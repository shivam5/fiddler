import torch

def custom_routing_function(hidden_states: torch.Tensor,
                            router_logits: torch.Tensor,
                            topk: int,
                            renormalize: bool):
    # Determine if it's prefill or decode phase based on sequence length
    # Prefill phase has sequence length 512, decode phase has sequence length 128
    is_prefill = hidden_states.shape[0] == 512
    profile_complete = True  # We'll assume profiling is complete
    
    # Default parameters
    num_total_experts = 8  # Mixtral has 8 experts
    num_experts_per_token = 2  # We use top-2 experts
    num_experts_to_keep = 8  # Default to keeping all experts
    min_experts = 2  # Minimum number of experts to keep
    policy = "do-nothing"  # Default policy
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

        if policy == "simple":
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
            beta = 0.5  # Default beta
            alpha = 2.0  # Default alpha

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
    
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    
    return topk_weights, topk_ids

def extract_voting_and_mask(topk_ids, gating_output, num_experts_to_drop):
    # Count votes for each expert
    expert_votes = torch.zeros(gating_output.shape[1], device=gating_output.device)
    for i in range(topk_ids.shape[1]):
        expert_votes.scatter_add_(0, topk_ids[:, i], torch.ones_like(topk_ids[:, i]))
    
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
        expert_votes.scatter_add_(0, topk_ids[:, i], topk_weights[:, i])
    
    # Calculate threshold based on percentile
    threshold = torch.quantile(expert_votes, threshold_percentile)
    
    # Select experts above threshold
    selected_experts = (expert_votes > threshold).nonzero().squeeze()
    num_experts_to_keep = max(min_experts, len(selected_experts))
    
    # If we have more experts than needed, take the top ones
    if len(selected_experts) > num_experts_to_keep:
        selected_experts = selected_experts[torch.argsort(expert_votes[selected_experts], descending=True)[:num_experts_to_keep]]
    
    return num_experts_to_keep, len(selected_experts)

def optimize_expert_selection_parameterized(topk_ids, topk_weights, gating_output, num_experts, min_experts, topk, beta, alpha):
    # Similar to optimize_expert_selection but with beta and alpha parameters
    expert_votes = torch.zeros(num_experts, device=gating_output.device)
    for i in range(topk_ids.shape[1]):
        expert_votes.scatter_add_(0, topk_ids[:, i], topk_weights[:, i])
    
    # Use beta to scale the threshold
    threshold = beta * torch.mean(expert_votes)
    
    # Use alpha to adjust the selection
    selected_experts = (expert_votes > threshold).nonzero().squeeze()
    num_experts_to_keep = max(min_experts, len(selected_experts))
    
    if len(selected_experts) > num_experts_to_keep:
        selected_experts = selected_experts[torch.argsort(expert_votes[selected_experts], descending=True)[:num_experts_to_keep]]
    
    return num_experts_to_keep, len(selected_experts)

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