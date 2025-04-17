import torch

def fused_topk(hidden_states: torch.Tensor,
               gating_output: torch.Tensor,
               topk: int,
               renormalize: bool):
    assert hidden_states.shape[0] == gating_output.shape[0]
    probs = torch.softmax(gating_output, dim=-1)
    topk_weights, topk_ids = torch.topk(probs, k=topk, dim=-1, sorted=False)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids

def extract_voting_and_mask(topk_ids: torch.Tensor,
                            gating_output: torch.Tensor,
                            num_experts_to_drop: int):
    num_experts = gating_output.size(1)
    expert_votes = torch.zeros(num_experts, device=topk_ids.device, dtype=torch.float16)
    flat_ids = topk_ids.reshape(-1)
    expert_votes.index_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.float16))
    _, drop_indices = torch.topk(expert_votes, k=num_experts_to_drop, largest=False)
    gating_output.index_fill_(1, drop_indices, -65504.0)
    return gating_output, drop_indices

def extract_important(topk_weights: torch.Tensor, threshold_percentile: float, epsilon=1e-8):
    top_half = topk_weights[:, : (topk_weights.size(1) // 2)].sum(dim=1)
    bottom_half = topk_weights[:, (topk_weights.size(1) // 2):].sum(dim=1)
    relative_diff = (top_half - bottom_half) / (top_half + epsilon)
    mask = relative_diff > threshold_percentile
    return mask, mask.sum().item()

def apply_expert_mask(gating_output: torch.Tensor, experts_to_drop_mask: torch.Tensor):
    mask_val = torch.finfo(gating_output.dtype).min
    gating_output.masked_fill_(experts_to_drop_mask.unsqueeze(0), mask_val)
    return gating_output

def optimize_expert_selection(topk_ids, topk_weights, gating_output,
                              topk, num_experts, threshold_percentile=0.5, min_experts=2):
    mask, _ = extract_important(topk_weights, threshold_percentile)
    topk_half = topk // 2
    expert_ids = topk_ids[:, :topk_half]
    weights = mask.unsqueeze(1).expand(-1, expert_ids.size(1)).to(torch.int32)
    flat_ids = expert_ids.reshape(-1)
    flat_weights = weights.reshape(-1)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    counts.scatter_add_(0, flat_ids, flat_weights)
    keep_mask = counts > 0
    keep_mask[:min_experts] = True
    drop_mask = ~keep_mask
    gating_output = apply_expert_mask(gating_output, drop_mask)
    return keep_mask.sum().item(), torch.unique(topk_ids).shape[0]

def optimize_expert_selection_parameterized(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gating_output: torch.Tensor,
    num_experts: int,
    min_experts: int = 2,
    topk: int = 6,
    beta: float = 0.5,
    alpha: float = 0.5,
):
    # Estimate number of unique experts without using torch.unique
    # Create a one-hot encoding of expert presence
    expert_presence = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    flat_ids = topk_ids.reshape(-1)
    ones = torch.ones_like(flat_ids, dtype=expert_presence.dtype)
    expert_presence.scatter_add_(0, flat_ids, ones)
    # Count experts with at least one occurrence
    num_unique_experts = (expert_presence > 0).sum().item()
    
    # Rest of the function remains the same
    B, K = topk_ids.shape
    confidence = extract_important_graded(topk_weights)
    alpha_val = 0
    keep_counts = (confidence * K * beta).floor().to(torch.int64) + alpha_val
    pos = torch.arange(K, device=topk_ids.device).expand(B, K)
    keep_mask = pos < keep_counts.unsqueeze(-1)
    kept_ids = torch.where(keep_mask, topk_ids, -1)
    flat_ids = kept_ids.reshape(-1)
    shifted = flat_ids + 1
    counts_plus1 = torch.zeros(num_experts + 1, dtype=torch.int32, device=topk_ids.device)
    ones = torch.ones_like(shifted, dtype=counts_plus1.dtype)
    counts_plus1.scatter_add_(0, shifted.to(torch.int64), ones)
    experts_to_keep_mask = counts_plus1[1:] > 0
    num_kept_total = experts_to_keep_mask.sum()
    gating_output = apply_expert_mask(gating_output, experts_to_drop_mask=~experts_to_keep_mask)
    return num_kept_total.item(), num_unique_experts

def extract_important_graded(topk_weights: torch.Tensor, epsilon: float = 1e-8):
    mid = topk_weights.size(1) // 2
    top_half_sum = topk_weights[:, :mid].sum(dim=1)
    bottom_half_sum = topk_weights[:, mid:].sum(dim=1)
    confidence = (top_half_sum - bottom_half_sum) / (top_half_sum + epsilon)
    return confidence

def rot_pref(gating_output: torch.Tensor, rank1: int, rank2: int):
    _, indices = torch.topk(gating_output, 6, dim=1)
    gating_output.scatter_(1, indices[:, rank1].unsqueeze(1), -65504.0)
    gating_output.scatter_(1, indices[:, rank2].unsqueeze(1), -65504.0)
    return gating_output

def extract_important_based_on_policy(policy: str, topk_weights: torch.Tensor,
                                      threshold_percentile: float = 0.5, topk: int = 6):
    if policy == "variance":
        return torch.var(topk_weights, dim=1)
    if policy == "relative_diff":
        top = topk_weights[:, :topk // 2].sum(dim=1)
        bot = topk_weights[:, topk // 2:].sum(dim=1)
        return (top - bot) / (top + 1e-8)
    if policy == "simple_diff":
        return (topk_weights[:, 0] - topk_weights[:, 1]) / (topk_weights[:, 0] + 1e-8)

def rotate_based_on_confidence(gating_output: torch.Tensor, topk_weights: torch.Tensor,
                                threshold_percentile: float, policy: str, quartile: str):
    confidence = extract_important_based_on_policy(policy, topk_weights, threshold_percentile, topk_weights.size(1))
    k = int(0.9 * confidence.size(0))
    if quartile == "keep_low_confidence":
        idx = torch.topk(confidence, k, largest=True).indices
    elif quartile == "keep_high_confidence":
        idx = torch.topk(confidence, k, largest=False).indices
    else:
        gating_output = rot_pref(gating_output, 0, 1)
        return
    gating_output[idx] = rot_pref(gating_output[idx], 1, 2)

# === Main function ===
def custom_routing_function(hidden_states: torch.Tensor,
                            router_logits: torch.Tensor,
                            topk: int,
                            renormalize: bool,
                            policy: str = "advanced_parametrized",
                            layer_idx: int = 0,
                            use_compile: bool = False,
                            is_expert_in_gpu_fn = None,
                            gpu_boost_factor: float = 5.0):
    is_prefill = hidden_states.shape[1] == 512

    # Replace with config or args
    num_total_experts = router_logits.size(-1)
    num_experts_to_keep = 4
    num_experts_per_token = topk
    min_experts = 2
    threshold_percentile = 0.5
    beta = 0.5
    alpha = 0.5
    rank1, rank2 = 1, 2
    confidence_policy = "simple_diff"
    quartile = "keep_high_confidence"

    if policy == "do-nothing" or is_prefill:
        return fused_topk(hidden_states, router_logits, topk, renormalize) + (num_total_experts,)

    topk_weights, topk_ids = fused_topk(hidden_states, router_logits, topk, renormalize=True)

    if policy == "simple":
        num_to_drop = num_total_experts - num_experts_to_keep
        masked_logits, _ = extract_voting_and_mask(topk_ids, router_logits, num_to_drop)
        topk_weights, topk_ids = fused_topk(hidden_states, masked_logits, topk, renormalize)

    elif policy == "advanced":
        masked_logits = router_logits
        num_experts_to_keep, _ = optimize_expert_selection(
            topk_ids, topk_weights, masked_logits, topk=num_experts_per_token,
            num_experts=num_total_experts, min_experts=min_experts,
            threshold_percentile=threshold_percentile
        )
        topk_weights, topk_ids = fused_topk(hidden_states, masked_logits, topk, renormalize)

    elif policy == "advanced_parametrized":
        masked_logits = router_logits
        num_experts_to_keep, _ = optimize_expert_selection_parameterized(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            gating_output=masked_logits,
            num_experts=num_total_experts,
            min_experts=min_experts,
            topk=num_experts_per_token,
            beta=beta,
            alpha=alpha,
        )
        topk_weights, topk_ids = fused_topk(hidden_states, masked_logits, topk, renormalize)

    elif policy == "rotate":
        rot_pref(router_logits, rank1, rank2)
        topk_weights, topk_ids = fused_topk(hidden_states, router_logits, topk, renormalize)
        num_experts_to_keep = num_total_experts - 2

    elif policy == "rotate_based_on_confidence":
        rotate_based_on_confidence(router_logits, topk_weights, threshold_percentile, confidence_policy, quartile)
        topk_weights, topk_ids = fused_topk(hidden_states, router_logits, topk, renormalize)

    return topk_weights, topk_ids, num_experts_to_keep
