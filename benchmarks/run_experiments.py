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

def run_experiment(batch_size, routing_policy, input_token=512, output_token=128, num_samples=3):
    """Run a single experiment with the given parameters"""
    
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Construct output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{results_dir}/bs{batch_size}_policy_{routing_policy}_{timestamp}.json"
    
    # Construct command
    cmd = [
        "python", "latency.py",
        "--batch_size", str(batch_size),
        "--routing_policy", routing_policy,
        "--input_token", str(input_token),
        "--output_token", str(output_token),
        "--num_samples", str(num_samples),
        "--output", output_file,
        "--skip_versioning"  # Skip automatic versioning after each run
    ]
    
    print(f"Running experiment: batch_size={batch_size}, policy={routing_policy}")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    subprocess.run(cmd, check=True)
    
    # Return the output file
    return output_file

def generate_plots(result_files, output_dir="plots"):
    """Generate plots from the experiment results"""
    
    # Create plots directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all results
    all_results = []
    for result_file in result_files:
        with open(result_file, 'r') as f:
            data = json.load(f)
            policy = data["config"]["routing_policy"]
            batch_size = data["config"]["batch_size"]
            
            # Add summary data to results
            summary = data["summary"]
            summary["policy"] = policy
            summary["batch_size"] = batch_size
            
            # Add expert utilization data if available
            if "results" in data and len(data["results"]) > 0:
                if "expert_utilization" in data["results"][0]:
                    for result in data["results"]:
                        if "expert_utilization" in result:
                            util = result["expert_utilization"]
                            if "avg_experts_per_layer" in util:
                                summary["avg_experts_per_layer"] = util["avg_experts_per_layer"]
                                summary["avg_gpu_experts_per_layer"] = util["avg_gpu_experts_per_layer"]
                                summary["avg_cpu_experts_per_layer"] = util["avg_cpu_experts_per_layer"]
            
            all_results.append(summary)
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(all_results)
    
    # 1. Plot latency vs batch size for different policies
    plt.figure(figsize=(12, 8))
    for policy, group in df.groupby("policy"):
        plt.plot(group["batch_size"], group["avg_decode_time"], marker='o', label=policy)
    
    plt.xlabel("Batch Size")
    plt.ylabel("Decode Time (s)")
    plt.title("Decode Latency vs Batch Size for Different Routing Policies")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/latency_vs_batch_size.png", dpi=300)
    
    # 2. Plot throughput improvement
    plt.figure(figsize=(12, 8))
    for policy, group in df.groupby("policy"):
        if policy == "do-nothing":
            continue
        
        # For each batch size, calculate improvement compared to baseline
        improvements = []
        batch_sizes = []
        
        for bs in sorted(df["batch_size"].unique()):
            baseline = df[(df["policy"] == "do-nothing") & (df["batch_size"] == bs)]["tokens_per_second"].values
            if len(baseline) == 0:
                continue
                
            current = df[(df["policy"] == policy) & (df["batch_size"] == bs)]["tokens_per_second"].values
            if len(current) == 0:
                continue
                
            improvement = (current[0] / baseline[0] - 1) * 100  # % improvement
            improvements.append(improvement)
            batch_sizes.append(bs)
        
        plt.plot(batch_sizes, improvements, marker='o', label=policy)
    
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput Improvement (%)")
    plt.title("Throughput Improvement vs Batch Size for Different Routing Policies")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/throughput_improvement.png", dpi=300)
    
    # 3. Plot GPU expert utilization
    plt.figure(figsize=(12, 8))
    for policy, group in df.groupby("policy"):
        plt.plot(group["batch_size"], group["avg_gpu_expert_percentage"] * 100, marker='o', label=policy)
    
    plt.xlabel("Batch Size")
    plt.ylabel("GPU Expert Utilization (%)")
    plt.title("GPU Expert Utilization vs Batch Size for Different Routing Policies")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/gpu_expert_utilization.png", dpi=300)
    
    # 4. Plot experts used per layer for different policies
    if "avg_experts_per_layer" in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Group by policy and get average
        policy_expert_usage = df.groupby("policy")[
            ["avg_experts_per_layer", "avg_gpu_experts_per_layer", "avg_cpu_experts_per_layer"]
        ].mean().reset_index()
        
        # Sort by avg_experts_per_layer
        policy_expert_usage = policy_expert_usage.sort_values("avg_experts_per_layer", ascending=False)
        
        # Create bar plot
        policies = policy_expert_usage["policy"]
        x = np.arange(len(policies))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x - width, policy_expert_usage["avg_experts_per_layer"], width, label="Total Experts")
        ax.bar(x, policy_expert_usage["avg_gpu_experts_per_layer"], width, label="GPU Experts")
        ax.bar(x + width, policy_expert_usage["avg_cpu_experts_per_layer"], width, label="CPU Experts")
        
        ax.set_ylabel("Avg Experts Per Layer")
        ax.set_title("Expert Utilization by Routing Policy")
        ax.set_xticks(x)
        ax.set_xticklabels(policies)
        ax.legend()
        ax.grid(True, axis='y')
        
        # Add a horizontal line at 8 (total experts)
        ax.axhline(y=8, color='r', linestyle='--', alpha=0.5, label="Total Experts Available")
        
        plt.savefig(f"{output_dir}/expert_utilization_by_policy.png", dpi=300)
    
    # 5. Bar chart comparing policies for a specific batch size
    plt.figure(figsize=(12, 8))
    
    # Select a batch size (e.g., the maximum one)
    max_batch_size = df["batch_size"].max()
    batch_df = df[df["batch_size"] == max_batch_size]
    
    policies = batch_df["policy"].tolist()
    decode_times = batch_df["avg_decode_time"].tolist()
    
    plt.bar(policies, decode_times)
    plt.xlabel("Routing Policy")
    plt.ylabel("Decode Time (s)")
    plt.title(f"Decode Time Comparison for Batch Size {max_batch_size}")
    plt.grid(True, axis='y')
    plt.savefig(f"{output_dir}/policy_comparison_bs{max_batch_size}.png", dpi=300)
    
    # 6. Save summary table as CSV
    df.to_csv(f"{output_dir}/summary_results.csv", index=False)
    
    print(f"Plots and summary saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run experiments for different MoE routing policies")
    # parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8], 
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[16, 32], 
                        help="Batch sizes to test")
    parser.add_argument("--policies", type=str, nargs="+", 
                        default=["do-nothing", "gpu_only", "simple", "advanced", "rotate"],
                        # default=["do-nothing", "simple", "advanced", "rotate"],
                        help="Routing policies to test")
    parser.add_argument("--input_token", type=int, default=512, 
                        help="Number of input tokens")
    parser.add_argument("--output_token", type=int, default=128, 
                        help="Number of output tokens")
    parser.add_argument("--num_samples", type=int, default=4, 
                        help="Number of samples per configuration")
    
    args = parser.parse_args()
    
    result_files = []
    
    # Run experiments for each combination of batch size and policy
    for batch_size in args.batch_sizes:
        for policy in args.policies:
            result_file = run_experiment(
                batch_size=batch_size,
                routing_policy=policy,
                input_token=args.input_token,
                output_token=args.output_token,
                num_samples=args.num_samples
            )
            result_files.append(result_file)
    
    # Generate plots from results
    generate_plots(result_files)
    
    # Version all results at once after all experiments are complete
    print("\nVersioning all results together...")
    version_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "version_results.sh")
    if os.path.exists(version_script) and os.access(version_script, os.X_OK):
        subprocess.run([version_script], check=True)
        print("Results versioned successfully.")
    else:
        print("Skipping versioning (version_results.sh not found or not executable)")

if __name__ == "__main__":
    main() 