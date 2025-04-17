import torch
from torch import Tensor
from typing import Tuple, Optional

@torch.jit.script
def fused_topk(
    hidden_states: Tensor,
    gating_output: Tensor,
    topk: int,
    renormalize: bool
) -> Tuple[Tensor, Tensor]:
    # CUDA graph-friendly topk implementation
    M, _ = hidden_states.shape
    
    # Using stable tensor operations
    weights = torch.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(weights, topk, dim=-1)
    
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    
    return topk_weights.to(hidden_states.dtype), topk_ids


def custom_routing_function(
    hidden_states: Tensor,
    router_logits: Tensor,
    topk: int,
    renormalize: bool,
    layer_idx: int = 0,
    num_total_experts: int = 8,
    policy: str = "do-nothing",
    gpu_boost_factor: float = 5.0,
    min_experts: int = 2,
    gpu_expert_mask: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, int]:
    # Static shape assertions for graph compatibility
    B, K = router_logits.shape
    assert K == num_total_experts, f"Expert count mismatch: got {K}, expected {num_total_experts}"
    
    # Use provided gpu_expert_mask or create a default one
    if gpu_expert_mask is None:
        # Default assumption: first half of experts are on GPU
        gpu_expert_mask = torch.zeros(num_total_experts, 
                                     device=router_logits.device,
                                     dtype=torch.bool)
        gpu_expert_mask[:num_total_experts//2] = True
    
    # Clone router logits to avoid modifying the input
    masked_logits = router_logits.clone()
    
    # Apply policy-specific modifications
    if policy == "gpu_only":
        # Static mask application for gpu_only policy
        large_negative = torch.tensor(-1e10, device=router_logits.device, dtype=router_logits.dtype)
        masked_logits = torch.where(
            gpu_expert_mask.unsqueeze(0).to(router_logits.device), 
            masked_logits,
            large_negative
        )
        
        # Compute routing weights with masked logits
        topk_weights, topk_ids = fused_topk(
            hidden_states, masked_logits, topk, renormalize
        )
        
        # Count GPU experts
        num_experts_to_keep = gpu_expert_mask.sum().int().item()
        
        return topk_weights, topk_ids, num_experts_to_keep
        
    elif policy == "gpu_boosted":
        # Static boost factor application for gpu_boosted policy
        boost_mask = torch.ones_like(masked_logits)
        boost_mask = boost_mask + (gpu_boost_factor - 1.0) * gpu_expert_mask.float().unsqueeze(0).to(boost_mask.device)
        
        # Apply the boost to the logits in one operation
        masked_logits = masked_logits * boost_mask
        
    # For all other policies, do standard topk
    topk_weights, topk_ids = fused_topk(
        hidden_states, masked_logits, topk, renormalize
    )
    
    # Default num_experts_to_keep for non-gpu_only policies
    num_experts_to_keep = num_total_experts
    
    return topk_weights, topk_ids, num_experts_to_keep


# Legacy support for backward compatibility
def extract_voting_and_mask(topk_ids, gating_output, num_experts_to_drop):
    # Count votes for each expert
    expert_votes = torch.zeros(gating_output.shape[1], device=gating_output.device)
    for i in range(topk_ids.shape[1]):
        ones = torch.ones_like(topk_ids[:, i], dtype=expert_votes.dtype)
        expert_votes.scatter_add_(0, topk_ids[:, i], ones)
    
    # Get least voted experts
    _, drop_indices = torch.topk(expert_votes, num_experts_to_drop, largest=False)
    
    # Mask out dropped experts
    mask = torch.ones_like(gating_output)
    mask[:, drop_indices] = 0
    masked_output = gating_output * mask
    
    return masked_output, drop_indices

def optimize_expert_selection(topk_ids, topk_weights, gating_output, num_experts, threshold_percentile, min_experts, topk):
    # Count votes for each expert
    expert_votes = torch.zeros(num_experts, device=gating_output.device)
    for i in range(topk_ids.shape[1]):
        # Ensure consistent dtype for scatter_add_
        weights = topk_weights[:, i].to(expert_votes.dtype)
        expert_votes.scatter_add_(0, topk_ids[:, i], weights)
    
    # Calculate threshold based on percentile
    threshold = torch.quantile(expert_votes, threshold_percentile)
    
    # Select experts above threshold
    selected_experts = (expert_votes > threshold).nonzero().squeeze()
    num_experts_to_keep = max(min_experts, len(selected_experts))
    
    # If we have more experts than needed, take the top ones
    if len(selected_experts) > num_experts_to_keep:
        selected_experts = selected_experts[torch.argsort(expert_votes[selected_experts], descending=True)[:num_experts_to_keep]]
    
    return num_experts_to_keep, len(selected_experts)

def extract_important_graded(topk_weights, epsilon=1e-8):
    """
    Returns a continuous confidence/importance score for every token.
    Compares top-half vs bottom-half weights.
    
    Args:
        topk_weights (Tensor): (batch_size, topk)
        epsilon (float): Numerical stabilizer
        
    Returns:
        Tensor: (batch_size,) confidence scores in [-1, 1]
    """
    mid = topk_weights.shape[1] // 2
    top_half_sum = topk_weights[:, :mid].sum(dim=1)        # (B,)
    bottom_half_sum = topk_weights[:, mid:].sum(dim=1)     # (B,)
    confidence = (top_half_sum - bottom_half_sum) / (top_half_sum + epsilon)
    
    return confidence

def optimize_expert_selection_parameterized(topk_ids, topk_weights, gating_output, num_experts, min_experts, topk, beta=0.7, alpha=0.0):
    """
    Parameterized expert selection based on token confidence.
    Formula: keep_counts = floor(confidence * K * beta) + alpha
    
    Args:
        topk_ids: Selected expert IDs (batch_size, topk)
        topk_weights: Weights for selected experts (batch_size, topk)
        gating_output: Full gating output
        num_experts: Total number of experts
        min_experts: Minimum experts to keep
        topk: Top-k value used for expert selection
        beta: Scaling factor (default 0.5)
        alpha: Base value (default 1.0)
        
    Returns:
        Tuple of (num_experts_to_keep, num_unique_experts)
    """
    # Calculate initial number of unique experts
    num_unique_experts = len(torch.unique(topk_ids))
    device = topk_ids.device
    
    # Calculate confidence scores for each token
    confidence = extract_important_graded(topk_weights)  # (batch_size,)
    
    # Calculate how many experts to keep per token
    # Formula: keep_counts = floor(confidence * K * beta) + alpha
    keep_counts = (confidence * topk * beta).floor().to(torch.int64) + alpha
    keep_counts = keep_counts.clamp(min=1, max=topk)  # Ensure at least 1 expert and at most topk
    
    # Create mask for positions to keep
    batch_size = topk_ids.shape[0]
    positions = torch.arange(topk, device=device).expand(batch_size, topk)  # (batch_size, topk)
    keep_mask = positions < keep_counts.unsqueeze(-1)  # (batch_size, topk)
    
    # Determine which experts to keep
    kept_ids = torch.where(keep_mask, topk_ids, torch.tensor(-1, device=device))  # Use -1 for masked positions
    flat_ids = kept_ids.reshape(-1)  # Flatten to (batch_size * topk)
    
    # Shift IDs by +1 so -1 maps to 0 (dummy bucket)
    shifted_ids = flat_ids + 1
    
    # Count occurrences of each expert
    counts = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    ones = torch.ones_like(shifted_ids, dtype=counts.dtype)
    counts.scatter_add_(0, shifted_ids.to(torch.int64), ones)
    
    # Experts to keep are those with count > 0 (excluding dummy bucket)
    experts_to_keep_mask = counts[1:] > 0  # (num_experts,)
    num_experts_to_keep = experts_to_keep_mask.sum().item()
    
    # Ensure minimum number of experts
    if num_experts_to_keep < min_experts:
        # If we have fewer experts than minimum, keep the experts with highest counts
        expert_counts = counts[1:]  # Remove dummy bucket
        _, top_experts = torch.topk(expert_counts, min_experts)
        new_mask = torch.zeros_like(experts_to_keep_mask)
        new_mask.scatter_(0, top_experts, 1)
        experts_to_keep_mask = new_mask
        num_experts_to_keep = min_experts
    
    # Apply mask to gating output
    for i in range(num_experts):
        if not experts_to_keep_mask[i]:
            # Zero out logits for experts not to keep
            gating_output[:, i] = torch.tensor(float('-inf'), device=device)
    
    return num_experts_to_keep, num_unique_experts

def rot_pref(gating_output, rank1, rank2):
    # Rotate preferences between two ranks
    mask = torch.ones_like(gating_output)
    mask[:, rank1] = 0
    mask[:, rank2] = 0
    gating_output = gating_output * mask

def rotate_based_on_confidence(gating_output, topk_weights, threshold_percentile, confidence_policy, quartile):
    # Calculate confidence scores
    if confidence_policy == "mean":
        confidence = torch.mean(topk_weights, dim=0)
    elif confidence_policy == "max":
        confidence = torch.max(topk_weights, dim=0)[0]
    else:
        confidence = torch.median(topk_weights, dim=0)[0]
    
    # Calculate threshold
    threshold = torch.quantile(confidence, quartile)
    
    # Mask out low confidence experts
    mask = (confidence > threshold).float()
    gating_output = gating_output * mask.unsqueeze(0)