import copy
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers


class FiddlerMixtral:
    def __init__(self, args):
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")
        self.model = transformers.MixtralForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            device_map='cpu',
            use_cache=True,
        )
        self.lm_head = self.model.lm_head
        self.model = self.model.model
        self.expert_placeholder = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0
        self.cpu_offload = args.cpu_offload
        self.beam_width = args.beam_num
        self.routing_policy = args.routing_policy
        self.n_layer = len(self.model.layers)
        self.n_expert = len(self.model.layers[0].block_sparse_moe.experts)
       
        # Store gpu_boost_factor if provided
        self.gpu_boost_factor = getattr(args, 'gpu_boost_factor', 5.0)

        # TODO: find this value based on device config
        self.latency_cpu = 7
        self.latency_gpu = 70

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        self.bring_non_expert_to_gpu()

        # 0: CPU, 1: GPU
        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)
        n_expert_on_gpu = self.calc_n_expert_on_gpu()
        print(
            f"Number of experts on GPU: {n_expert_on_gpu}/{self.n_layer * self.n_expert}"
        )
        print(
            f"Number of layers: {self.n_layer}, number of experts per layer: {self.n_expert}"
        )

        self.set_expert_loc(n_expert_on_gpu)
        # print(self.expert_loc)

        self.bring_expert_to_gpu()

        print("Model is ready.")

    def bring_non_expert_to_gpu(self):
        """Bring non-expert layers to GPU"""
        self.lm_head.to(self.dev)
        self.model.embed_tokens.to(self.dev)
        self.model.norm.to(self.dev)
        for i in range(len(self.model.layers)):
            self.model.layers[i].self_attn.to(self.dev)
            self.model.layers[i].input_layernorm.to(self.dev)
            self.model.layers[i].block_sparse_moe.gate.to(self.dev)
            self.model.layers[i].post_attention_layernorm.to(self.dev)
            # only model.layers[i].block_sparse_moe.experts is on CPU

    def set_expert_loc(self, n_expert_on_gpu, popular_experts=None):
        """Set the location of experts"""
        if popular_experts is None:
            # list of (i_layer, i_expert) in the order of popularity
            # determined based on profile
            popular_experts = [
                (9, 5),
                (11, 2),
                (10, 4),
                (28, 0),
                (13, 1),
                (17, 7),
                (12, 1),
                (8, 6),
                (16, 1),
                (9, 0),
                (14, 5),
                (19, 5),
                (26, 2),
                (30, 7),
                (7, 1),
                (3, 7),
                (23, 4),
                (22, 1),
                (29, 3),
                (1, 5),
                (13, 0),
                (5, 1),
                (18, 0),
                (4, 7),
                (10, 3),
                (1, 2),
                (3, 0),
                (8, 3),
                (11, 0),
                (11, 5),
                (11, 1),
                (31, 4),
                (21, 0),
                (25, 1),
                (15, 5),
                (22, 4),
                (27, 5),
                (16, 7),
                (15, 1),
                (13, 2),
                (15, 4),
                (21, 1),
                (27, 7),
                (9, 7),
                (7, 4),
                (31, 5),
                (2, 1),
                (11, 6),
                (12, 3),
                (2, 4),
                (24, 2),
                (28, 2),
                (0, 2),
                (30, 2),
                (6, 0),
                (6, 7),
                (15, 6),
                (6, 2),
                (14, 2),
                (2, 0),
                (17, 2),
                (19, 2),
                (24, 0),
                (10, 0),
                (19, 4),
                (1, 4),
                (26, 3),
                (31, 7),
                (17, 6),
                (25, 3),
                (12, 6),
                (0, 0),
                (26, 0),
                (29, 7),
                (27, 2),
                (19, 6),
                (5, 0),
                (18, 2),
                (20, 1),
                (12, 4),
                (17, 5),
                (5, 4),
                (30, 6),
                (20, 5),
                (24, 6),
                (25, 2),
                (28, 4),
                (4, 6),
                (7, 2),
                (20, 3),
                (23, 2),
                (8, 4),
                (30, 0),
                (3, 4),
                (12, 5),
                (23, 7),
                (1, 7),
                (22, 5),
                (18, 4),
                (31, 0),
                (17, 0),
                (0, 5),
                (14, 6),
                (0, 3),
                (15, 7),
                (5, 6),
                (4, 4),
                (24, 7),
                (31, 1),
                (27, 6),
                (22, 2),
                (14, 1),
                (1, 0),
                (29, 1),
                (21, 3),
                (25, 7),
                (22, 3),
                (7, 3),
                (2, 6),
                (29, 5),
                (28, 3),
                (6, 6),
                (7, 5),
                (5, 7),
                (8, 5),
                (20, 4),
                (21, 5),
                (18, 7),
                (27, 0),
                (16, 0),
                (24, 5),
                (12, 2),
                (2, 2),
                (24, 3),
                (4, 1),
                (29, 0),
                (3, 1),
                (21, 6),
                (10, 2),
                (20, 7),
                (19, 0),
                (26, 7),
                (20, 6),
                (23, 3),
                (4, 3),
                (30, 1),
                (1, 6),
                (29, 2),
                (30, 3),
                (0, 6),
                (8, 1),
                (25, 6),
                (29, 4),
                (16, 2),
                (23, 1),
                (26, 1),
                (26, 6),
                (16, 4),
                (2, 5),
                (0, 4),
                (7, 6),
                (14, 4),
                (3, 6),
                (20, 0),
                (18, 3),
                (4, 5),
                (17, 4),
                (0, 1),
                (16, 5),
                (19, 3),
                (23, 0),
                (30, 4),
                (20, 2),
                (13, 6),
                (18, 6),
                (15, 2),
                (3, 5),
                (22, 0),
                (10, 1),
                (9, 6),
                (10, 5),
                (25, 4),
                (9, 2),
                (18, 1),
                (6, 4),
                (4, 2),
                (23, 5),
                (6, 5),
                (21, 2),
                (5, 5),
                (6, 1),
                (26, 5),
                (12, 0),
                (25, 0),
                (4, 0),
                (14, 0),
                (16, 6),
                (31, 2),
                (8, 0),
                (21, 7),
                (14, 3),
                (31, 6),
                (28, 1),
                (5, 3),
                (23, 6),
                (6, 3),
                (18, 5),
                (25, 5),
                (27, 1),
                (11, 7),
                (11, 4),
                (24, 1),
                (0, 7),
                (8, 7),
                (13, 3),
                (21, 4),
                (27, 4),
                (13, 7),
                (3, 2),
                (9, 1),
                (2, 7),
                (7, 0),
                (2, 3),
                (28, 5),
                (27, 3),
                (15, 0),
                (24, 4),
                (5, 2),
                (22, 6),
                (3, 3),
                (28, 6),
                (14, 7),
                (13, 4),
                (28, 7),
                (22, 7),
                (13, 5),
                (19, 1),
                (26, 4),
                (1, 1),
                (17, 1),
                (16, 3),
                (10, 7),
                (29, 6),
                (19, 7),
                (31, 3),
                (7, 7),
                (1, 3),
                (8, 2),
                (9, 4),
                (17, 3),
                (30, 5),
                (15, 3),
                (9, 3),
                (10, 6),
                (12, 7),
                (11, 3),
            ]

        for i in range(n_expert_on_gpu):
            i_layer, i_expert = popular_experts[i]
            self.expert_loc[i_layer, i_expert] = 1

    def bring_expert_to_gpu(self):
        """Bring part of expert layers to GPU"""
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                if self.is_expert_in_gpu(i, j):
                    self.model.layers[i].block_sparse_moe.experts[j].to(self.dev)

    def is_expert_in_gpu(self, i_layer, i_expert):
        """Determine if the expert is in GPU"""
        return self.expert_loc[i_layer, i_expert] == 1

    def calc_n_expert_on_gpu(self):
        """Get the number of experts that we can put on GPU"""
        # get the number of parameters of one expert
        n_param = sum(
            p.numel()
            for p in self.model.layers[0].block_sparse_moe.experts[0].parameters()
        )
        # get the amount of free memory on GPU
        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        free_mem = total_mem * 0.95 - torch.cuda.memory_allocated(self.dev) # TODO: magic number
        return int((free_mem) // (n_param * 2))

    def initial_beam_tensor(self, input_tensor):
        # transpose tensor of shape (beam_width, seq_len, beam_width) to (beam_width, 1) properly
        assert input_tensor.shape[-1] == self.beam_width
        input_tensor = input_tensor[:, -1]
        row_idx = torch.tensor(
            [i * self.beam_width for i in range(input_tensor.shape[0] // self.beam_width)]
        )
        output_tensor = input_tensor[row_idx].view(-1, 1)
        return output_tensor

    def generate(self, texts=None, output_token=20, input_token=None):
        torch.set_num_threads(16) # TODO: set appropriately
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0
        
        # Handle single text case
        if isinstance(texts, str):
            texts = [texts]
        
        input_ids_list = []
        position_ids_list = []
        for text in texts:
            input_ids, position_ids = self.tokenize(text)
            if input_token is not None:
                input_ids = input_ids[:, :input_token]
                position_ids = position_ids[:, :input_token]
            input_ids_list.append(input_ids)
            position_ids_list.append(position_ids)
        
        # Stack all inputs
        input_ids = torch.cat(input_ids_list, dim=0)
        position_ids = torch.cat(position_ids_list, dim=0)

        # Log initial info before timing
        print(f"Starting generation of {output_token} tokens for batch size {input_ids.shape[0]}...")
        
        tick = time.time()
        is_decode = False
        prefill_time, decode_time = 0, 0
        decode_strings = ["" for _ in range(input_ids.shape[0])]
        search_start = False
        probs = torch.full((input_ids.shape[0], 1), 1.0)

        # print(f"Starting generation of {output_token} tokens for batch size {input_ids.shape[0]}...")
        for i_token in range(output_token):
            # if i_token % 5 == 0:  # Log every 5 tokens
            #     print(f"Generating token {i_token + 1}/{output_token}")
            #     if is_decode:
            #         print(f"Current outputs: {[s[:50] + '...' for s in decode_strings]}")

            # if self.beam_width == 1:
            #     print([self.tokenizer.decode(ids) for ids in input_ids])
            # if is_decode:
            #     for i in range(input_ids.shape[0]):
            #         decode_strings[i] += " " + self.tokenizer.decode(input_ids[i, :])

            logits = self.mixtral_forward(input_ids, position_ids, is_decode)

            logits = logits.to("cpu")
            # logits.shape: (batch_size, seq_len, vocab_size)

            # normalize logits
            logits = F.softmax(logits, dim=-1)

            # greedy search:
            # output = torch.argmax(logits, dim=-1)

            # beam_search:
            self.past_key_values_length += logits.shape[1]
            if search_start:
                new_probs, output = torch.topk(logits, 1, dim=-1)
                new_probs = new_probs[:, -1].flatten().view(-1, 1)
            else:
                new_probs, output = torch.topk(logits, self.beam_width, dim=-1)
                new_probs = self.initial_beam_tensor(new_probs)
                output = self.initial_beam_tensor(output)
                search_start = True
            # new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)
            probs = probs * new_probs

            input_ids = output[:, -1].flatten().view(-1, 1).to(self.dev)
            # input_ids.shape: (batch_size, seq_len=1)

            position_ids = (
                torch.arange(
                    self.past_key_values_length,
                    self.past_key_values_length + 1,
                    dtype=torch.long,
                    device=self.dev,
                )
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1)
            )
            # position_ids.shape: (batch_size, 1)
            if not is_decode:
                prefill_time += time.time() - tick
                tick = time.time()
            is_decode = True
        decode_time = time.time() - tick
        probs = probs.view(-1, self.beam_width)
        max_ids = torch.argmax(probs, dim=-1)

        # Log completion info after timing
        print("\nGeneration complete!")
        print("--------------------")
        print(f"Inputs: {texts}")
        print(f"Outputs: {[decode_strings[i] for i in max_ids]}")
        print(f"Prefill time: {prefill_time:.2f}s")
        print(f"Decode time: {decode_time:.2f}s")
        print(f"Expert hit rate: {self.cnt_expert_hit / self.cnt_expert_all:.2%}")

        return (
            prefill_time,
            decode_time,
            self.cnt_expert_hit / self.cnt_expert_all,
        )

    def tokenize(self, text):
        input_ids = []
        encodings = self.tokenizer(text, return_tensors="pt")
        input_id = encodings.input_ids.to(self.dev)
        for i in range(self.beam_width):
            input_ids.append(input_id[0])
        
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.dev)

        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=self.dev
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])

        return input_ids, position_ids

    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids, is_decode):
        hidden_dim = self.model.config.hidden_size
        inps = input_ids.to(self.dev)
        inps = self.model.embed_tokens(inps)

        total_experts_processed = 0
        cpu_experts_processed = 0
        gpu_experts_processed = 0
        
        # Track expert utilization statistics
        expert_stats = {
            "total": {"all": 0, "gpu": 0, "cpu": 0},
            "by_layer": {},
            "decode_step": is_decode,  # Flag to indicate if this is a decode step
            "tokens_per_batch": inps.shape[0],  # Number of tokens being processed
            "unique_experts_per_batch": {}  # Track unique experts used across the entire batch
        }

        # Count experts activated per token in batch
        total_experts_per_token = 0
        total_tokens = 0
        # Track unique experts used per batch across all layers
        batch_unique_experts_total = 0

        for i_layer, layer in enumerate(self.model.layers):
            # Initialize layer stats
            layer_stats = {
                "unique_experts_used": 0, 
                "experts_on_gpu": 0, 
                "experts_on_cpu": 0, 
                "tokens_processed": 0, 
                "experts_per_token": 0,
                "unique_experts_per_batch": 0  # New metric to track unique experts per batch
            }
            expert_stats["by_layer"][i_layer] = layer_stats
            
            original_inps_shape = inps.shape

            inps_residual = inps
            inps = layer.input_layernorm(inps)
            inps, self_attn_weights, present_key_value = layer.self_attn(
                inps,
                position_ids=position_ids,
                past_key_value=self.past_key_value,
                use_cache=True,
            )
            inps = inps_residual + inps
            inps_residual = inps
            inps = layer.post_attention_layernorm(inps)
            inps = inps.view(-1, hidden_dim)
            
            router_logits = layer.block_sparse_moe.gate(inps)
            
            # Use custom routing if specified
            if hasattr(self, 'routing_policy'):
                from fiddler.custom_routing import custom_routing_function
                routing_weights, selected_experts, num_experts_to_keep = custom_routing_function(
                    hidden_states=inps,
                    router_logits=router_logits,
                    topk=2,  # We use top-2 experts
                    renormalize=True,
                    model=self,
                    i_layer=i_layer,
                    policy=self.routing_policy
                )
                # Ensure correct dtypes
                if selected_experts.dtype != torch.int64:
                    selected_experts = selected_experts.to(torch.int64)
                routing_weights = routing_weights.to(inps.dtype)
            else:
                routing_weights = F.softmax(router_logits, dim=1)
                routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            # Track all unique experts across the entire batch for this layer
            unique_experts_batch = set()
            for i in range(selected_experts.shape[0]):  # Iterate through batch
                for j in range(selected_experts.shape[1]):  # Iterate through top-k experts
                    unique_experts_batch.add(selected_experts[i, j].item())
            layer_stats["unique_experts_per_batch"] = len(unique_experts_batch)
            batch_unique_experts_total += len(unique_experts_batch)

            # intermediate variable to store the output of experts
            inps_after_experts = torch.zeros_like(inps, device=self.dev)
            experts = layer.block_sparse_moe.experts

            if self.cpu_offload == 0:
                # baseline: do everything at GPU
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=8
                ).permute(2, 1, 0)

                # Track which experts are actually used in this layer
                experts_used = set()
                experts_on_gpu = set()
                experts_on_cpu = set()
                
                # Count how many tokens are processed
                num_tokens_processed = inps.shape[0]
                layer_stats["tokens_processed"] = num_tokens_processed
                total_tokens += num_tokens_processed
                
                # Track experts activated per token
                experts_activated = 0

                for i_expert in range(len(experts)):
                    is_cuda = self.is_expert_in_gpu(i_layer, i_expert)
                    idx, top_2 = torch.where(expert_mask[i_expert])

                    if top_2.shape[0] == 0:
                        continue
                    
                    # Count how many tokens use this expert
                    experts_activated += top_2.shape[0]

                    # This expert is actually used
                    experts_used.add(i_expert)
                    if is_cuda:
                        experts_on_gpu.add(i_expert)
                    else:
                        experts_on_cpu.add(i_expert)

                    total_experts_processed += 1
                    if is_cuda:
                        gpu_experts_processed += 1
                    else:
                        cpu_experts_processed += 1

                    top_2_list = top_2.tolist()
                    idx_list = idx.tolist()

                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    if not is_cuda:
                        self.expert_placeholder.load_state_dict(
                            experts[i_expert].state_dict()
                        )
                        current_state = self.expert_placeholder(
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    else:
                        current_state = experts[i_expert](
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    inps_after_experts.index_add_(
                        0, top_2, current_state.to(inps_after_experts.dtype)
                    )

                    if not is_cuda:
                        experts[i_expert] = experts[i_expert].to("cpu")

                # Calculate experts per token
                experts_per_token = experts_activated / max(1, num_tokens_processed)
                layer_stats["experts_per_token"] = experts_per_token
                total_experts_per_token += experts_per_token
                
                # Update layer statistics
                layer_stats["unique_experts_used"] = len(experts_used)
                layer_stats["experts_on_gpu"] = len(experts_on_gpu)
                layer_stats["experts_on_cpu"] = len(experts_on_cpu)
                layer_stats["experts_used"] = list(experts_used)
                layer_stats["gpu_experts"] = list(experts_on_gpu)
                layer_stats["cpu_experts"] = list(experts_on_cpu)
                
                # Update total statistics
                expert_stats["total"]["all"] += len(experts_used)
                expert_stats["total"]["gpu"] += len(experts_on_gpu)
                expert_stats["total"]["cpu"] += len(experts_on_cpu)
            else:
                # prefill stage with offloading
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=8
                ).permute(2, 1, 0)

                # Track which experts are needed
                experts_used = set()
                experts_on_gpu = set()
                experts_on_cpu = set()
                
                # Count how many tokens are processed
                num_tokens_processed = inps.shape[0]
                layer_stats["tokens_processed"] = num_tokens_processed
                total_tokens += num_tokens_processed
                
                # Track experts activated per token
                experts_activated = 0
                
                # first, calculate the number of tokens for each expert
                idxs, top_2s = [], []
                cost_per_expert = np.zeros(
                    (len(experts), 2), dtype=float
                )  # 0: CPU, 1: GPU
                for i_expert in range(len(experts)):
                    idx, top_2 = torch.where(expert_mask[i_expert])
                    idxs.append(idx)
                    top_2s.append(top_2)
                    
                    # Count how many tokens use this expert
                    experts_activated += top_2.shape[0]
                    
                    # expected latency at CPU: number of token * cost_at_cpu
                    # expected latency at GPU: cost_at_gpu (constant)
                    cost_per_expert[i_expert, 0] = top_2.shape[0] * self.latency_cpu
                    cost_per_expert[i_expert, 1] = self.latency_gpu
                    
                    # Track if this expert is used
                    if top_2.shape[0] > 0:
                        experts_used.add(i_expert)
                        if self.is_expert_in_gpu(i_layer, i_expert):
                            experts_on_gpu.add(i_expert)
                        else:
                            experts_on_cpu.add(i_expert)
                    
                    if self.is_expert_in_gpu(i_layer, i_expert):
                        # if the expert is in GPU, the latency at GPU is
                        # approximately 0
                        cost_per_expert[i_expert, 1] = 0
                        self.cnt_expert_hit += top_2.shape[0]
                    self.cnt_expert_all += top_2.shape[0]
                
                # Calculate experts per token
                experts_per_token = experts_activated / max(1, num_tokens_processed)
                layer_stats["experts_per_token"] = experts_per_token
                total_experts_per_token += experts_per_token
                
                # Update layer statistics
                layer_stats["unique_experts_used"] = len(experts_used)
                layer_stats["experts_on_gpu"] = len(experts_on_gpu)
                layer_stats["experts_on_cpu"] = len(experts_on_cpu)
                layer_stats["experts_used"] = list(experts_used)
                layer_stats["gpu_experts"] = list(experts_on_gpu)
                layer_stats["cpu_experts"] = list(experts_on_cpu)
                
                # We already calculated unique_experts_per_batch above, so no need to do it again
                
                # Update total statistics
                expert_stats["total"]["all"] += len(experts_used)
                expert_stats["total"]["gpu"] += len(experts_on_gpu)
                expert_stats["total"]["cpu"] += len(experts_on_cpu)
                
                # second, partition experts processing between CPU and GPU so that we can minimize:
                # max(sum of cost at CPU, sum of cost at GPU)
                # greedy algorithm is just as there are only 8 experts for Mixtral
                best_config = -1
                best_cost = float("inf")
                for config in range(1 << len(experts)):
                    sum_cost = 0
                    for i_expert in range(len(experts)):
                        if (config >> i_expert) & 1:
                            sum_cost += cost_per_expert[i_expert, 0]
                        else:
                            sum_cost += cost_per_expert[i_expert, 1]
                    if sum_cost < best_cost:
                        best_cost = sum_cost
                        best_config = config

                # then, we can offload the experts according to the best
                # configuration
                cpu_experts = []
                gpu_experts = []
                for i_expert in range(8):
                    if (best_config >> i_expert) & 1:
                        cpu_experts.append(i_expert)
                    else:
                        gpu_experts.append(i_expert)

                total_experts_processed += len(experts)
                cpu_experts_processed += len(cpu_experts)
                gpu_experts_processed += len(gpu_experts)

                for i_expert in gpu_experts:
                    top_2_list = top_2s[i_expert].tolist()
                    idx_list = idxs[i_expert].tolist()
                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    if self.is_expert_in_gpu(i_layer, i_expert):
                        current_state = experts[i_expert](
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    else:
                        self.expert_placeholder.load_state_dict(
                            experts[i_expert].state_dict()
                        )
                        current_state = self.expert_placeholder(
                            current_state, routing_weights[top_2_list, idx_list, None]
                        )
                    inps_after_experts.index_add_(
                        0,
                        top_2s[i_expert].to(self.dev, non_blocking=True),
                        current_state.to(self.dev, non_blocking=True),
                    )

                for i_expert in cpu_experts:
                    top_2_list = top_2s[i_expert].tolist()
                    idx_list = idxs[i_expert].tolist()
                    current_state = inps[None, top_2_list].reshape(-1, hidden_dim)
                    current_state = self.run_expert_at_cpu(
                        i_layer,
                        i_expert,
                        current_state.to("cpu"),
                        routing_weights[top_2_list, idx_list, None].to("cpu"),
                    )
                    inps_after_experts.index_add_(
                        0,
                        top_2s[i_expert].to(self.dev, non_blocking=True),
                        current_state.to(self.dev, non_blocking=True),
                    )

            # addition because there's residual connection over moe layer
            inps = inps_residual + inps_after_experts.reshape(original_inps_shape)

            # end of one layer

        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        self.present_key_value = present_key_value
        
        # Update object attributes for metrics collection
        self.total_experts_processed = total_experts_processed
        self.gpu_experts_processed = gpu_experts_processed
        self.cpu_experts_processed = cpu_experts_processed
        self.expert_stats = expert_stats
                
        # Calculate average experts used per layer
        expert_stats["avg_experts_per_layer"] = expert_stats["total"]["all"] / len(self.model.layers)
        expert_stats["avg_gpu_experts_per_layer"] = expert_stats["total"]["gpu"] / len(self.model.layers)
        expert_stats["avg_cpu_experts_per_layer"] = expert_stats["total"]["cpu"] / len(self.model.layers)
        
        # Calculate average experts per token per layer
        if total_tokens > 0:
            expert_stats["avg_experts_per_token"] = total_experts_per_token / len(self.model.layers)
        else:
            expert_stats["avg_experts_per_token"] = 0
        
        # Calculate average unique experts per batch across all layers
        expert_stats["avg_unique_experts_per_batch"] = batch_unique_experts_total / len(self.model.layers)
                
        return lm_logis

    def run_expert_at_cpu(self, i_layer, i_expert, inps, routing_weights):
        """Run the expert at CPU"""
        return self.model.layers[i_layer].block_sparse_moe.experts[i_expert](
            inps, routing_weights
        )