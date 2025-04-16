"""Microbenchmarking for CPU offloading"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

sys.path.append("../src")
from fiddler import FiddlerMixtral

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model path. default `mistralai/Mixtral-8x7B-v0.1`.",
    )
    parser.add_argument(
        "--cpu-offload",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: exeute at GPU (baseline), 1: offload to CPU.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size for inference.",
    )
    parser.add_argument("--beam_num", type=int, default=1, help="Beam search number.")
    parser.add_argument(
        "--routing_policy",
        type=str,
        default="do-nothing",
        choices=["do-nothing", "simple", "advanced", "advanced_parametrized", "rotate", "rotate_based_on_confidence"],
        help="Routing policy to use for expert selection.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Output file to store results.",
    )
    parser.add_argument(
        "--input_token",
        type=int,
        default=512,
        help="Number of input tokens.",
    )
    parser.add_argument(
        "--output_token",
        type=int,
        default=128,
        help="Number of output tokens to generate.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to run for each configuration.",
    )

    args = parser.parse_args()

    # Use absolute path for the ShareGPT file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_json = os.path.join(base_dir, "mixtral_offloading", "Mixtral-8x7B-Instruct-v0.1", "ShareGPT_V3_unfiltered_cleaned_split.json")
    
    if not os.path.exists(path_json):
        print(f"Error: Could not find ShareGPT file at {path_json}")
        print("Please ensure the file exists at the correct location.")
        sys.exit(1)

    with open(path_json, "r") as f:
        data = json.load(f)

    texts = []
    for d in data:
        if len(d["conversations"]) == 0:
            continue
        # the input of the first round
        texts.append(" ".join(d["conversations"][0]["value"].split()))

    random.seed(0)
    random.shuffle(texts)
    
    model = FiddlerMixtral(args)
    
    # Metrics to collect
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": args.model,
            "cpu_offload": args.cpu_offload,
            "batch_size": args.batch_size,
            "beam_num": args.beam_num,
            "routing_policy": args.routing_policy,
            "input_token": args.input_token,
            "output_token": args.output_token,
        },
        "results": []
    }
    
    # Run the experiments
    input_token = args.input_token
    output_token = args.output_token
    
    idx_text = 0
    for sample_idx in range(args.num_samples):
        print(f"Running sample {sample_idx+1}/{args.num_samples}")
        
        # Collect batch_size texts
        batch_texts = []
        while len(batch_texts) < args.batch_size:
            text = texts[idx_text % len(texts)]
            idx_text += 1
            if len(text.split()) >= input_token:
                # enough input length
                batch_texts.append(text)
        
        # Add custom attributes to track detailed expert metrics
        model.gpu_experts_processed = 0
        model.cpu_experts_processed = 0
        model.total_experts_processed = 0
        
        # Start timing
        start_time = time.time()
        prefill_time, decode_time, hit_rate = model.generate(
            batch_texts, output_token=output_token, input_token=input_token
        )
        total_time = time.time() - start_time
        
        # Record results
        metrics["results"].append({
            "sample_idx": sample_idx,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": total_time,
            "hit_rate": hit_rate,
            "throughput": output_token / decode_time,
            "total_experts_processed": getattr(model, "total_experts_processed", 0),
            "gpu_experts_processed": getattr(model, "gpu_experts_processed", 0),
            "cpu_experts_processed": getattr(model, "cpu_experts_processed", 0),
            "gpu_expert_percentage": getattr(model, "gpu_experts_processed", 0) / max(1, getattr(model, "total_experts_processed", 1))
        })
        
        print(f"Sample {sample_idx+1} complete: Prefill: {prefill_time:.2f}s, Decode: {decode_time:.2f}s, Hit rate: {hit_rate:.2%}")
    
    # Calculate averages
    num_samples = len(metrics["results"])
    if num_samples > 0:
        avg_prefill_time = sum(r["prefill_time"] for r in metrics["results"]) / num_samples
        avg_decode_time = sum(r["decode_time"] for r in metrics["results"]) / num_samples
        avg_hit_rate = sum(r["hit_rate"] for r in metrics["results"]) / num_samples
        avg_throughput = sum(r["throughput"] for r in metrics["results"]) / num_samples
        avg_gpu_expert_pct = sum(r["gpu_expert_percentage"] for r in metrics["results"]) / num_samples
        
        metrics["summary"] = {
            "avg_prefill_time": avg_prefill_time,
            "avg_decode_time": avg_decode_time,
            "avg_total_time": avg_prefill_time + avg_decode_time,
            "avg_hit_rate": avg_hit_rate,
            "avg_throughput": avg_throughput,
            "avg_gpu_expert_percentage": avg_gpu_expert_pct,
            "tokens_per_second": output_token * args.batch_size / avg_decode_time
        }
        
        print("\nSummary:")
        print(f"Routing Policy: {args.routing_policy}, Batch Size: {args.batch_size}")
        print(f"Avg Prefill Time: {avg_prefill_time:.2f}s")
        print(f"Avg Decode Time: {avg_decode_time:.2f}s")
        print(f"Avg Hit Rate: {avg_hit_rate:.2%}")
        print(f"Tokens/sec: {output_token * args.batch_size / avg_decode_time:.2f}")
        print(f"GPU Expert Usage: {avg_gpu_expert_pct:.2%}")
    
    # Save results to file
    result_file = args.output
    os.makedirs(os.path.dirname(os.path.abspath(result_file)), exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {result_file}")
