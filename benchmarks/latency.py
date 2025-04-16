"""Microbenchmarking for CPU offloading"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
import subprocess

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
        choices=["do-nothing", "simple", "advanced", "advanced_parametrized", "rotate", "rotate_based_on_confidence", "gpu_only"],
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
    for batch_num in range(args.num_samples // args.batch_size):
        print(f"Running batch {batch_num+1}/{args.num_samples // args.batch_size}")
        
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
        
        # Start timing - we'll use the times returned by the model itself
        # which only measure the actual computation time, not the logging
        prefill_time, decode_time, hit_rate = model.generate(
            batch_texts, output_token=output_token, input_token=input_token
        )
        
        # Record results
        metrics["results"].append({
            "batch_num": batch_num,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "hit_rate": hit_rate,
            "throughput": output_token / decode_time,
            "total_experts_processed": getattr(model, "total_experts_processed", 0),
            "gpu_experts_processed": getattr(model, "gpu_experts_processed", 0),
            "cpu_experts_processed": getattr(model, "cpu_experts_processed", 0),
            "gpu_expert_percentage": getattr(model, "gpu_experts_processed", 0) / max(1, getattr(model, "total_experts_processed", 1)),
            "expert_utilization": getattr(model, "expert_stats", {})
        })
        
        # Print expert utilization stats
        if hasattr(model, "expert_stats"):
            expert_stats = model.expert_stats
            print("\nExpert Utilization Statistics:")
            print(f"Routing Policy: {args.routing_policy}")
            print(f"Batch Size: {args.batch_size}")
            print(f"Phase: {'Decode' if expert_stats.get('decode_step', False) else 'Prefill'}")
            print(f"Tokens in this batch: {expert_stats.get('tokens_per_batch', 0)}")
            
            if "avg_experts_per_token" in expert_stats:
                print(f"\nExperts per token: {expert_stats['avg_experts_per_token']:.2f}")
                print(f"  (This should be close to 2.0 for top-2 routing)")
            
            if "avg_unique_experts_per_batch" in expert_stats:
                print(f"\nUnique experts per batch: {expert_stats['avg_unique_experts_per_batch']:.2f}")
                print(f"  (This is the average number of unique experts used across the entire batch)")
            
            print(f"\nAverage unique experts used per layer: {expert_stats['avg_experts_per_layer']:.2f} out of 8")
            print(f"  - On GPU: {expert_stats['avg_gpu_experts_per_layer']:.2f}")
            print(f"  - On CPU: {expert_stats['avg_cpu_experts_per_layer']:.2f}")
            
            # Print the layers with highest/lowest expert utilization
            layer_experts_per_batch = [(layer_idx, stats.get("unique_experts_per_batch", 0)) 
                                for layer_idx, stats in expert_stats["by_layer"].items()]
            layer_experts_per_batch.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nLayers with highest unique experts per batch:")
            for layer_idx, unique_experts in layer_experts_per_batch[:3]:
                layer_stats = expert_stats["by_layer"][layer_idx]
                print(f"  Layer {layer_idx}: {unique_experts} unique experts across batch, " + 
                      f"{layer_stats['unique_experts_used']} unique experts used")
                
            print(f"\nLayers with lowest unique experts per batch:")
            for layer_idx, unique_experts in layer_experts_per_batch[-3:]:
                layer_stats = expert_stats["by_layer"][layer_idx]
                print(f"  Layer {layer_idx}: {unique_experts} unique experts across batch, " + 
                      f"{layer_stats['unique_experts_used']} unique experts used")
        
        print(f"Batch {batch_num+1} complete: Prefill: {prefill_time:.2f}s, Decode: {decode_time:.2f}s, Hit rate: {hit_rate:.2%}")
    
    # Calculate averages
    num_samples = len(metrics["results"])
    if num_samples > 0:
        avg_prefill_time = sum(r["prefill_time"] for r in metrics["results"]) / num_samples
        avg_decode_time = sum(r["decode_time"] for r in metrics["results"]) / num_samples
        avg_hit_rate = sum(r["hit_rate"] for r in metrics["results"]) / num_samples
        avg_throughput = sum(r["throughput"] for r in metrics["results"]) / num_samples
        avg_gpu_expert_pct = sum(r["gpu_expert_percentage"] for r in metrics["results"]) / num_samples
        
        # Calculate average unique experts per batch if available
        avg_unique_experts_per_batch = 0
        count_with_metric = 0
        for r in metrics["results"]:
            expert_utilization = r.get("expert_utilization", {})
            if "avg_unique_experts_per_batch" in expert_utilization:
                avg_unique_experts_per_batch += expert_utilization["avg_unique_experts_per_batch"]
                count_with_metric += 1
        
        if count_with_metric > 0:
            avg_unique_experts_per_batch /= count_with_metric
        
        metrics["summary"] = {
            "avg_prefill_time": avg_prefill_time,
            "avg_decode_time": avg_decode_time,
            "avg_total_time": avg_prefill_time + avg_decode_time,
            "avg_hit_rate": avg_hit_rate,
            "avg_throughput": avg_throughput,
            "avg_gpu_expert_percentage": avg_gpu_expert_pct,
            "tokens_per_second": output_token * args.batch_size / avg_decode_time,
            "avg_unique_experts_per_batch": avg_unique_experts_per_batch
        }
        
        print("\nSummary:")
        print(f"Routing Policy: {args.routing_policy}, Batch Size: {args.batch_size}")
        print(f"Avg Prefill Time: {avg_prefill_time:.2f}s")
        print(f"Avg Decode Time: {avg_decode_time:.2f}s")
        print(f"Avg Hit Rate: {avg_hit_rate:.2%}")
        print(f"Tokens/sec: {output_token * args.batch_size / avg_decode_time:.2f}")
        print(f"GPU Expert Usage: {avg_gpu_expert_pct:.2%}")
        if count_with_metric > 0:
            print(f"Avg Unique Experts Per Batch: {avg_unique_experts_per_batch:.2f}")
    
    # Save results to file
    result_file = args.output
    os.makedirs(os.path.dirname(os.path.abspath(result_file)), exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {result_file}")
    
    # Run versioning script if it exists
    version_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "version_results.sh")
    if os.path.exists(version_script) and os.access(version_script, os.X_OK):
        print("\nVersioning results...")
        subprocess.run([version_script], check=True)
        print("Results versioned successfully.")
    else:
        print("\nSkipping versioning (version_results.sh not found or not executable)")
