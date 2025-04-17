#!/usr/bin/env python3
"""
Runner script for measuring latency improvements of different MoE routing policies.
"""

import os
import subprocess
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime

# current direcotory + runs + run_2
run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "run_8")


def run_experiment(batch_size, routing_policy, input_token=512, output_token=128, num_samples=3, gpu_boost_factor=None, use_compile=False):
    """Run a single experiment with the given parameters"""
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Construct output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    policy_name = routing_policy
    compile_suffix = "_compiled" if use_compile else ""
    
    if routing_policy == "gpu_boosted" and gpu_boost_factor is not None:
        policy_name = f"{routing_policy}_theta{gpu_boost_factor}"
    
    output_file = f"{results_dir}/bs{batch_size}_policy_{policy_name}{compile_suffix}_{timestamp}.json"
    
    # Construct command
    cmd = [
        "python", "latency.py",
        "--batch_size", str(batch_size),
        "--routing_policy", routing_policy,
        "--input_token", str(input_token),
        "--output_token", str(output_token),
        "--num_samples", str(num_samples),
        "--output", output_file
    ]
    
    # Add GPU boost factor if specified
    if gpu_boost_factor is not None:
        cmd.extend(["--gpu_boost_factor", str(gpu_boost_factor)])
    
    # Add compile flag if specified
    if use_compile:
        cmd.append("--use_compile")
    
    print(f"Running experiment: batch_size={batch_size}, policy={policy_name}{compile_suffix}")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    subprocess.run(cmd, check=True)
    
    # Return the output file
    return output_file

def generate_plots(result_files, output_dir=os.path.join(run_dir, "plots")):
    """Generate plots from the experiment results"""
    
    # Create plots directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all results
    all_results = []
    for result_file in result_files:
        with open(result_file, 'r') as f:
            data = json.load(f)
            policy = data["config"]["routing_policy"]
            
            # Extract theta value if this is a gpu_boosted policy
            if "gpu_boost_factor" in data["config"] and policy == "gpu_boosted":
                policy = f"gpu_boosted_theta{data['config']['gpu_boost_factor']}"
            
            # Add compiled suffix if compilation was used
            if "use_compile" in data["config"] and data["config"]["use_compile"]:
                policy += "_compiled"
            
            batch_size = data["config"]["batch_size"]
            
            # Add summary data to results
            summary = data["summary"]
            summary["policy"] = policy
            summary["batch_size"] = batch_size
            
            # Add unique experts per batch if available
            if "avg_unique_experts_per_batch" in summary:
                summary["avg_unique_experts_per_batch"] = summary["avg_unique_experts_per_batch"]
            
            all_results.append(summary)
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(all_results)
    
    # Create separate dataframes for compiled and non-compiled results
    compiled_df = df[df["policy"].str.contains("_compiled")]
    noncompiled_df = df[~df["policy"].str.contains("_compiled")]
    
    # Create comparison plots if we have both compiled and non-compiled results
    if not compiled_df.empty and not noncompiled_df.empty:
        # Create a dataframe for showing speedups
        speedup_data = []
        
        for policy in noncompiled_df["policy"].unique():
            compiled_policy = f"{policy}_compiled"
            if compiled_policy not in compiled_df["policy"].values:
                continue
                
            for batch_size in noncompiled_df["batch_size"].unique():
                non_comp_time = noncompiled_df[(noncompiled_df["policy"] == policy) & 
                                              (noncompiled_df["batch_size"] == batch_size)]["avg_decode_time"].values
                
                comp_time = compiled_df[(compiled_df["policy"] == compiled_policy) & 
                                        (compiled_df["batch_size"] == batch_size)]["avg_decode_time"].values
                
                if len(non_comp_time) > 0 and len(comp_time) > 0:
                    speedup = non_comp_time[0] / comp_time[0]
                    speedup_data.append({
                        "policy": policy.replace("_compiled", ""),
                        "batch_size": batch_size,
                        "speedup": speedup
                    })
        
        if speedup_data:
            speedup_df = pd.DataFrame(speedup_data)
            
            # Plot speedup
            plt.figure(figsize=(14, 8))
            batch_sizes = speedup_df["batch_size"].unique()
            
            if len(batch_sizes) == 1:
                # Bar chart for single batch size
                batch_speedup_df = speedup_df[speedup_df["batch_size"] == batch_sizes[0]]
                batch_speedup_df = batch_speedup_df.sort_values("speedup", ascending=False)
                
                bars = plt.bar(batch_speedup_df["policy"], batch_speedup_df["speedup"])
                plt.axhline(y=1.0, color='r', linestyle='--')
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                            f'{height:.2f}x', ha='center', va='bottom')
            else:
                # Line chart for multiple batch sizes
                for policy, group in speedup_df.groupby("policy"):
                    sorted_group = group.sort_values("batch_size")
                    plt.plot(sorted_group["batch_size"], sorted_group["speedup"], marker='o', label=policy)
                plt.axhline(y=1.0, color='r', linestyle='--')
                plt.legend()
            
            plt.xlabel("Policy" if len(batch_sizes) == 1 else "Batch Size")
            plt.ylabel("Speedup (x)")
            plt.title(f"Speedup from Compilation (Batch Size {batch_sizes[0]})" if len(batch_sizes) == 1
                     else "Speedup from Compilation vs Batch Size")
            plt.grid(True)
            plt.savefig(f"{output_dir}/compilation_speedup.png", dpi=300)
    
    # Get unique batch sizes
    batch_sizes = df["batch_size"].unique()
    is_single_batch = len(batch_sizes) == 1
    
    # 1. Plot average decode time
    plt.figure(figsize=(14, 8))
    if is_single_batch:
        # Bar chart for single batch size
        batch_df = df.sort_values("avg_decode_time")
        plt.bar(batch_df["policy"], batch_df["avg_decode_time"])
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
    else:
        # Line chart for multiple batch sizes
        for policy, group in df.groupby("policy"):
            sorted_group = group.sort_values("batch_size")
            plt.plot(sorted_group["batch_size"], sorted_group["avg_decode_time"], marker='o', label=policy)
        plt.legend()
    
    plt.xlabel("Policy" if is_single_batch else "Batch Size")
    plt.ylabel("Average Decode Time (s)")
    plt.title("Average Decode Time by Policy" if is_single_batch else "Average Decode Time vs Batch Size")
    plt.grid(True)
    plt.savefig(f"{output_dir}/avg_decode_time.png", dpi=300)
    
    # 2. Plot average hit rate
    plt.figure(figsize=(14, 8))
    if is_single_batch:
        # Bar chart for single batch size
        batch_df = df.sort_values("avg_hit_rate", ascending=False)
        bars = plt.bar(batch_df["policy"], batch_df["avg_hit_rate"] * 100)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{height:.1f}%', ha='center', va='bottom')
    else:
        # Line chart for multiple batch sizes
        for policy, group in df.groupby("policy"):
            sorted_group = group.sort_values("batch_size")
            plt.plot(sorted_group["batch_size"], sorted_group["avg_hit_rate"] * 100, marker='o', label=policy)
        plt.legend()
    
    plt.xlabel("Policy" if is_single_batch else "Batch Size")
    plt.ylabel("Average Hit Rate (%)")
    plt.title("Average Hit Rate by Policy" if is_single_batch else "Average Hit Rate vs Batch Size")
    plt.grid(True)
    plt.savefig(f"{output_dir}/avg_hit_rate.png", dpi=300)
    
    # 3. Plot average unique experts per batch (if available)
    if "avg_unique_experts_per_batch" in df.columns:
        plt.figure(figsize=(14, 8))
        if is_single_batch:
            # Bar chart for single batch size
            if df["avg_unique_experts_per_batch"].notnull().any():
                batch_df = df.sort_values("avg_unique_experts_per_batch", ascending=False)
                batch_df = batch_df[batch_df["avg_unique_experts_per_batch"].notnull()]
                bars = plt.bar(batch_df["policy"], batch_df["avg_unique_experts_per_batch"])
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                             f'{height:.1f}', ha='center', va='bottom')
        else:
            # Line chart for multiple batch sizes
            for policy, group in df.groupby("policy"):
                sorted_group = group.sort_values("batch_size")
                if sorted_group["avg_unique_experts_per_batch"].notnull().any():
                    plt.plot(sorted_group["batch_size"], 
                             sorted_group["avg_unique_experts_per_batch"], 
                             marker='o', label=policy)
            plt.legend()
        
        plt.xlabel("Policy" if is_single_batch else "Batch Size")
        plt.ylabel("Avg Unique Experts Per Batch")
        plt.title("Average Unique Experts Per Batch by Policy" if is_single_batch 
                 else "Average Unique Experts Per Batch vs Batch Size")
        plt.grid(True)
        plt.savefig(f"{output_dir}/avg_unique_experts_per_batch.png", dpi=300)
    
    # Save summary table as CSV
    df.to_csv(f"{output_dir}/summary_results.csv", index=False)
    
    print(f"Plots and summary saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run experiments for different MoE routing policies")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[4], 
                        help="Batch sizes to test")
    parser.add_argument("--policies", type=str, nargs="+", 
                        default=["advanced_parametrized"],
                        # default=["advanced_parametrized", "do-nothing", "gpu_only"],
                        help="Routing policies to test")
    parser.add_argument("--run_gpu_boosted", action="store_true", default=False,
                        help="Run gpu_boosted policy with different theta values")
    parser.add_argument("--use_compile", action="store_true", default=True,
                        help="Run experiments with torch.compile enabled")
    parser.add_argument("--run_with_without_compile", action="store_true", default=False,
                        help="Run each experiment both with and without compilation")
    parser.add_argument("--input_token", type=int, default=512, 
                        help="Number of input tokens")
    parser.add_argument("--output_token", type=int, default=128, 
                        help="Number of output tokens")
    parser.add_argument("--num_samples", type=int, default=0, 
                        help="Number of samples per configuration (if 0, will match batch_size)")
    
    args = parser.parse_args()
    
    result_files = []
    
    # Run experiments for each combination of batch size and policy
    for batch_size in args.batch_sizes:
        # Set num_samples to match batch_size to run exactly 1 batch
        experiment_samples = batch_size if args.num_samples == 0 else args.num_samples
        
        compile_options = [False, True] if args.run_with_without_compile else [args.use_compile]
        
        for policy in args.policies:
            for use_compile in compile_options:
                result_file = run_experiment(
                    batch_size=batch_size,
                    routing_policy=policy,
                    input_token=args.input_token,
                    output_token=args.output_token,
                    num_samples=experiment_samples,
                    use_compile=use_compile
                )
                result_files.append(result_file)
        
        # Run gpu_boosted with different theta values
        if args.run_gpu_boosted:
            theta_values = [1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
            for theta in theta_values:
                for use_compile in compile_options:
                    result_file = run_experiment(
                        batch_size=batch_size,
                        routing_policy="gpu_boosted",
                        input_token=args.input_token,
                        output_token=args.output_token,
                        num_samples=experiment_samples,
                        gpu_boost_factor=theta,
                        use_compile=use_compile
                    )
                    result_files.append(result_file)
    
    # Generate plots from results
    generate_plots(result_files)

if __name__ == "__main__":
    main() 