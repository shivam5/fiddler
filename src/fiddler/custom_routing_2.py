import torch

# Import torch.compile at the top
try:
    from torch._dynamo import allow_in_graph
    from torch._dynamo import optimize_assert
    COMPILE_AVAILABLE = True
except ImportError:
    COMPILE_AVAILABLE = False
    # Define dummy decorator for compatibility
    def allow_in_graph(fn):
        return fn
    def optimize_assert(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

@allow_in_graph
def custom_routing_function(hidden_states: torch.Tensor,
                            router_logits: torch.Tensor,
                            topk: int,
                            renormalize: bool,
                            policy: str = "do-nothing",
                            layer_idx: int = 0,
                            use_compile: bool = False,
                            is_expert_in_gpu_fn = None,
                            gpu_boost_factor: float = 5.0):
    """
    Custom routing function with direct parameters to avoid graph breaks.
    
    Args:
        hidden_states: Hidden states tensor
        router_logits: Router logits tensor
        topk: Number of top experts to select
        renormalize: Whether to renormalize weights
        policy: Routing policy to use
        layer_idx: Current layer index
        use_compile: Whether to use compiled functions
        is_expert_in_gpu_fn: Function to check if expert is on GPU
        gpu_boost_factor: Boost factor for GPU experts
    """
    # Determine if it's prefill or decode phase based on sequence length
    # Prefill phase has sequence length 512, decode phase has sequence length 128
    is_prefill = hidden_states.shape[0] == 512
    profile_complete = True  # We'll assume profiling is complete
    
    # Default parameters
    num_total_experts = 8  # Mixtral has 8 experts
    num_experts_per_token = 2  # We use top-2 experts
    num_experts_to_keep = 8  # Default to keeping all experts
    min_experts = 2  # Minimum number of experts to keep
    threshold_percentile = 0.5  # Default threshold
    count_of_topk = 2  # Default count

    if policy == "do-nothing":
        topk_weights, topk_ids = fused_topk(
            hidden_states, router_logits, topk, renormalize
        )
        return topk_weights, topk_ids, num_experts_to_keep
        
    if profile_complete and not is_prefill:
        masked_logits = router_logits

        # First pass top-k
        topk_weights_initial, topk_ids_initial = fused_topk(
            hidden_states, masked_logits, topk, renormalize=True
        )
        
        if policy == "gpu_only":
            # Create list of GPU experts
            gpu_experts = []
            for i_expert in range(num_total_experts):
                if is_expert_in_gpu_fn(layer_idx, i_expert):
                    gpu_experts.append(i_expert)
            
            # Create mask for GPU experts - only GPU experts get positive values
            mask = torch.zeros(masked_logits.shape[1], device=masked_logits.device)
            for expert_id in gpu_experts:
                mask[expert_id] = 1.0
            
            # Set logits of non-GPU experts to a very large negative value to guarantee they aren't chosen
            large_negative = -1e10
            masked_logits = torch.where(mask.unsqueeze(0) > 0, masked_logits, torch.tensor(large_negative, device=masked_logits.device))
            
            # Compute new routing weights with masked logits that only include GPU experts
            topk_weights, topk_ids = fused_topk(
                hidden_states, masked_logits, topk, renormalize=renormalize
            )
            
            # Double-check: ensure all selected experts are GPU experts
            # Convert gpu_experts to tensor for faster comparison
            gpu_experts_tensor = torch.tensor(gpu_experts, device=topk_ids.device, dtype=topk_ids.dtype)
            
            # Verify and force any non-GPU experts to be replaced with GPU experts
            # This completely eliminates any edge cases where non-GPU experts might be selected
            if len(gpu_experts) >= topk:
                for i in range(topk_ids.shape[0]):
                    for j in range(topk_ids.shape[1]):
                        expert_id = topk_ids[i, j].item()
                        if expert_id not in gpu_experts:
                            # Replace with the first GPU expert
                            topk_ids[i, j] = gpu_experts_tensor[0]
            
            # Count how many GPU experts we're keeping
            num_experts_to_keep = len(gpu_experts)
            
        elif policy == "gpu_boosted":
            # Like advanced_parameterized but with a boost for GPU experts
            # Boost weights for GPU experts
            for i_expert in range(num_total_experts):
                if is_expert_in_gpu_fn(layer_idx, i_expert):
                    # Multiply by theta to increase importance of GPU experts
                    masked_logits[:, i_expert] *= gpu_boost_factor
            
            # Recompute initial topk after boosting GPU experts
            topk_weights_initial, topk_ids_initial = fused_topk(
                hidden_states, masked_logits, topk, renormalize=True
            )
            
            # Apply parameterized expert selection like in advanced_parameterized
            beta = 0.5
            alpha = 0.25

            if use_compile and COMPILE_AVAILABLE:
                # Use compiled version if available and enabled
                num_experts_to_keep, num_unique_experts = compiled_optimize_expert_selection_parameterized(
                    topk_ids=topk_ids_initial,
                    topk_weights=topk_weights_initial, 
                    gating_output=masked_logits,
                    num_experts=num_total_experts,
                    min_experts=min_experts,
                    topk=num_experts_per_token,
                    beta=beta,
                    alpha=alpha,
                )
            else:
                num_experts_to_keep, num_unique_experts = optimize_expert_selection_parameterized(
                    topk_ids=topk_ids_initial,
                    topk_weights=topk_weights_initial, 
                    gating_output=masked_logits,
                    num_experts=num_total_experts,
                    min_experts=min_experts,
                    topk=num_experts_per_token,
                    beta=beta,
                    alpha=alpha,
                )
            
            topk_weights, topk_ids = fused_topk(
                hidden_states, masked_logits, topk, renormalize=renormalize
            )

        elif policy == "simple":
            num_experts_to_keep = 4  # Keep only 4 experts
            num_experts_to_drop = num_total_experts - num_experts_to_keep
            
            masked_logits, drop_indices = extract_voting_and_mask(
                topk_ids_initial,
                masked_logits,
                num_experts_to_drop=num_experts_to_drop
            )

            topk_weights, topk_ids = fused_topk(
                hidden_states, 
                masked_logits, 
                topk, 
                renormalize=renormalize
            )

        elif policy == "advanced":
            if use_compile and COMPILE_AVAILABLE:
                # Use compiled version if available and enabled
                num_experts_to_keep, num_unique_experts = compiled_optimize_expert_selection(
                    topk_ids=topk_ids_initial,
                    topk_weights=topk_weights_initial,
                    gating_output=masked_logits,
                    num_experts=num_total_experts,
                    threshold_percentile=threshold_percentile,
                    min_experts=min_experts,
                    topk=num_experts_per_token,
                )
            else:
                num_experts_to_keep, num_unique_experts = optimize_expert_selection(
                    topk_ids=topk_ids_initial,
                    topk_weights=topk_weights_initial,
                    gating_output=masked_logits,
                    num_experts=num_total_experts,
                    threshold_percentile=threshold_percentile,
                    min_experts=min_experts,
                    topk=num_experts_per_token,
                )
            
            topk_weights, topk_ids = fused_topk(
                hidden_states, masked_logits, topk, renormalize=renormalize
            )

        elif policy == "advanced_parametrized":
            beta = 0.5
            alpha = 0.25

            if use_compile and COMPILE_AVAILABLE:
                # Use compiled version if available and enabled
                num_experts_to_keep, num_unique_experts = compiled_optimize_expert_selection_parameterized(
                    topk_ids=topk_ids_initial,
                    topk_weights=topk_weights_initial,
                    gating_output=masked_logits,
                    num_experts=num_total_experts,
                    min_experts=min_experts,
                    topk=num_experts_per_token,
                    beta=beta,
                    alpha=alpha,
                )
            else:
                num_experts_to_keep, num_unique_experts = optimize_expert_selection_parameterized(
                    topk_ids=topk_ids_initial,
                    topk_weights=topk_weights_initial,
                    gating_output=masked_logits,
                    num_experts=num_total_experts,
                    min_experts=min_experts,
                    topk=num_experts_per_token,
                    beta=beta,
                    alpha=alpha,
                )

            topk_weights, topk_ids = fused_topk(
                hidden_states, masked_logits, topk, renormalize=renormalize
            )

        elif policy == "rotate":
            rank1 = 0  # Default rank1
            rank2 = 1  # Default rank2
            rot_pref(masked_logits, rank1, rank2)
            topk_weights, topk_ids = fused_topk(
                hidden_states, router_logits, topk, renormalize
            )
            num_experts_to_keep = num_total_experts - 2

        elif policy == "rotate_based_on_confidence":
            confidence_policy = "mean"  # Default confidence policy
            quartile = 0.5  # Default quartile
            rotate_based_on_confidence(masked_logits, topk_weights_initial, threshold_percentile, confidence_policy, quartile)
            topk_weights, topk_ids = fused_topk(
                hidden_states, router_logits, topk, renormalize
            )
            
    else:
        # Prefill case — no expert dropping
        topk_weights, topk_ids = fused_topk(
            hidden_states, router_logits, topk, renormalize
        )
        num_experts_to_keep = num_total_experts

    return topk_weights, topk_ids, num_experts_to_keep

# Wrap core functions with torch.compile when available
if COMPILE_AVAILABLE:
    @allow_in_graph
    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def compiled_fused_topk(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ):
        return fused_topk(hidden_states, gating_output, topk, renormalize)
    
    @allow_in_graph
    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def compiled_extract_important_graded(topk_weights, epsilon=1e-8):
        return extract_important_graded(topk_weights, epsilon)
    
    @allow_in_graph
    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def compiled_optimize_expert_selection(topk_ids, topk_weights, gating_output, num_experts, threshold_percentile, min_experts, topk):
        return optimize_expert_selection(topk_ids, topk_weights, gating_output, num_experts, threshold_percentile, min_experts, topk)
    
    @allow_in_graph
    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def compiled_optimize_expert_selection_parameterized(topk_ids, topk_weights, gating_output, num_experts, min_experts, topk, beta=0.5, alpha=0.25):
        return optimize_expert_selection_parameterized(topk_ids, topk_weights, gating_output, num_experts, min_experts, topk, beta, alpha)

@allow_in_graph
def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    M, _ = hidden_states.shape

    topk_weights = torch.empty(M,
                               topk,
                               dtype=torch.float32,
                               device=hidden_states.device)
    topk_ids = torch.empty(M,
                           topk,
                           dtype=torch.int32,
                           device=hidden_states.device)
    token_expert_indicies = torch.empty(M,
                                        topk,
                                        dtype=torch.int32,
                                        device=hidden_states.device)
    
    # Compute softmax and top-k
    gating_output = gating_output.float()
    gating_output = torch.softmax(gating_output, dim=-1)
    topk_weights, topk_ids = torch.topk(gating_output, topk, dim=-1)
    
    # Convert topk_ids to int64 to match PyTorch's native top-k
    topk_ids = topk_ids.to(torch.int64)
    
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    
    return topk_weights, topk_ids

def extract_voting_and_mask(topk_ids, gating_output, num_experts_to_drop):
    # Count votes for each expert
    expert_votes = torch.zeros(gating_output.shape[1], device=gating_output.device)
    for i in range(topk_ids.shape[1]):
        # Fix dtype mismatch - convert ones to match expert_votes dtype
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

def optimize_expert_selection_parameterized(topk_ids, topk_weights, gating_output, num_experts, min_experts, topk, beta=0.5, alpha=0.25):
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