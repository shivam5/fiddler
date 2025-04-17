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
run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "run_11_cuda_graphs")


def run_experiment(batch_size, routing_policy, input_token=512, output_token=128, num_samples=3, gpu_boost_factor=None, use_cuda_graph=False):
    """Run a single experiment with the given parameters"""
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Construct output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    policy_name = routing_policy
    if routing_policy == "gpu_boosted" and gpu_boost_factor is not None:
        policy_name = f"{routing_policy}_theta{gpu_boost_factor}"
    
    output_file = f"{results_dir}/bs{batch_size}_policy_{policy_name}_{timestamp}.json"
    
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
    
    # Add CUDA graph option
    if use_cuda_graph:
        cmd.append("--use_cuda_graph")
    
    print(f"Running experiment: batch_size={batch_size}, policy={policy_name}")
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
                theta = data['config']['gpu_boost_factor']
                policy = f"gpu_boosted (θ={theta})"
            
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
    
    # Get unique batch sizes
    batch_sizes = df["batch_size"].unique()
    is_single_batch = len(batch_sizes) == 1
    
    # Sort policies for consistent ordering in plots
    # Put basic policies first, then gpu_boosted with different theta values in ascending order
    def policy_sort_key(policy_name):
        if "gpu_boosted" in policy_name and "θ=" in policy_name:
            # Extract theta value and sort by it
            try:
                theta = float(policy_name.split("θ=")[1].rstrip(")"))
                return (1, theta)  # Sort gpu_boosted policies after basic ones, by theta
            except:
                return (1, 999)  # Fallback
        else:
            # Basic policies
            policy_order = {
                "do-nothing": 0,
                "simple": 1,
                "advanced": 2,
                "advanced_parametrized": 3,
                "gpu_only": 4,
                "rotate": 5,
                "rotate_based_on_confidence": 6
            }
            return (0, policy_order.get(policy_name, 100))  # Basic policies first, ordered by importance
    
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
        # Sort policies for consistent colors and markers in the legend
        sorted_policies = sorted(df["policy"].unique(), key=policy_sort_key)
        
        # Predefined colors and markers for better visualization
        colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_policies)))
        markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h', 'X', 'd']
        
        for i, policy in enumerate(sorted_policies):
            group = df[df["policy"] == policy]
            sorted_group = group.sort_values("batch_size")
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            plt.plot(sorted_group["batch_size"], sorted_group["avg_decode_time"], 
                     marker=marker, color=color, label=policy, linewidth=2)
        
        plt.legend(loc='best', fontsize=10)
    
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
        # Sort policies for consistent colors and markers in the legend
        sorted_policies = sorted(df["policy"].unique(), key=policy_sort_key)
        
        # Predefined colors and markers for better visualization
        colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_policies)))
        markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h', 'X', 'd']
        
        for i, policy in enumerate(sorted_policies):
            group = df[df["policy"] == policy]
            sorted_group = group.sort_values("batch_size")
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            plt.plot(sorted_group["batch_size"], sorted_group["avg_hit_rate"] * 100, 
                     marker=marker, color=color, label=policy, linewidth=2)
        
        plt.legend(loc='best', fontsize=10)
    
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
            # Sort policies for consistent colors and markers in the legend
            sorted_policies = sorted(df["policy"].unique(), key=policy_sort_key)
            
            # Predefined colors and markers for better visualization
            colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_policies)))
            markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h', 'X', 'd']
            
            for i, policy in enumerate(sorted_policies):
                group = df[df["policy"] == policy]
                sorted_group = group.sort_values("batch_size")
                if sorted_group["avg_unique_experts_per_batch"].notnull().any():
                    color = colors[i % len(colors)]
                    marker = markers[i % len(markers)]
                    plt.plot(sorted_group["batch_size"], 
                             sorted_group["avg_unique_experts_per_batch"], 
                             marker=marker, color=color, label=policy, linewidth=2)
            
            plt.legend(loc='best', fontsize=10)
        
        plt.xlabel("Policy" if is_single_batch else "Batch Size")
        plt.ylabel("Avg Unique Experts Per Batch")
        plt.title("Average Unique Experts Per Batch by Policy" if is_single_batch 
                 else "Average Unique Experts Per Batch vs Batch Size")
        plt.grid(True)
        plt.savefig(f"{output_dir}/avg_unique_experts_per_batch.png", dpi=300)
    
    # 4. Create a dedicated plot for gpu_boosted with different theta values
    gpu_boosted_df = df[df["policy"].str.contains("gpu_boosted")]
    if not gpu_boosted_df.empty:
        # Extract theta values for x-axis
        gpu_boosted_df["theta"] = gpu_boosted_df["policy"].apply(
            lambda x: float(x.split("θ=")[1].rstrip(")")) if "θ=" in x else 0)
        
        plt.figure(figsize=(14, 8))
        # Plot hit rate vs theta
        for batch, group in gpu_boosted_df.groupby("batch_size"):
            sorted_group = group.sort_values("theta")
            plt.plot(sorted_group["theta"], sorted_group["avg_hit_rate"] * 100, 
                     marker='o', label=f"Batch Size {batch}", linewidth=2)
        
        plt.xlabel("GPU Boost Factor (θ)")
        plt.ylabel("Average Hit Rate (%)")
        plt.title("Effect of GPU Boost Factor (θ) on Hit Rate")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{output_dir}/gpu_boost_theta_comparison.png", dpi=300)
    
    # Save summary table as CSV
    df.to_csv(f"{output_dir}/summary_results.csv", index=False)
    
    print(f"Plots and summary saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run experiments for different MoE routing policies")
    # parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8], 
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[4], 
                        help="Batch sizes to test")
    parser.add_argument("--policies", type=str, nargs="+", 
                        default=["do-nothing", "advanced_parametrized", "gpu_only"],
                        help="Routing policies to test")
    parser.add_argument("--run_gpu_boosted", action="store_true", default=True,
                        help="Run gpu_boosted policy with different theta values")
    parser.add_argument("--theta_values", type=float, nargs="+", 
                        default=[1.2, 1.5, 2.0, 3.0, 5.0],
                        help="Theta values to test with gpu_boosted policy")
    parser.add_argument("--input_token", type=int, default=512, 
                        help="Number of input tokens")
    parser.add_argument("--output_token", type=int, default=128, 
                        help="Number of output tokens")
    parser.add_argument("--num_samples", type=int, default=0, 
                        help="Number of samples per configuration (if 0, will match batch_size)")
    parser.add_argument("--use_cuda_graph", action="store_true", default=False,
                        help="Use CUDA graph for experiments")
    
    args = parser.parse_args()
    
    result_files = []
    
    # Run experiments for each combination of batch size and policy
    for batch_size in args.batch_sizes:
        # Set num_samples to match batch_size to run exactly 1 batch
        experiment_samples = batch_size if args.num_samples == 0 else args.num_samples
        
        # Run regular policies
        for policy in args.policies:
            # Skip gpu_boosted if we're running it separately with different theta values
            if policy == "gpu_boosted" and args.run_gpu_boosted:
                continue
                
            result_file = run_experiment(
                batch_size=batch_size,
                routing_policy=policy,
                input_token=args.input_token,
                output_token=args.output_token,
                num_samples=experiment_samples,
                use_cuda_graph=args.use_cuda_graph
            )
            result_files.append(result_file)
        
        # Run gpu_boosted with different theta values
        if args.run_gpu_boosted:
            print(f"\nRunning gpu_boosted policy with {len(args.theta_values)} different theta values: {args.theta_values}")
            
            for theta in args.theta_values:
                print(f"\nTesting gpu_boosted with θ={theta}")
                result_file = run_experiment(
                    batch_size=batch_size,
                    routing_policy="gpu_boosted",
                    input_token=args.input_token,
                    output_token=args.output_token,
                    num_samples=experiment_samples,
                    gpu_boost_factor=theta,
                    use_cuda_graph=args.use_cuda_graph
                )
                result_files.append(result_file)
    
    # Generate plots from results
    generate_plots(result_files)

if __name__ == "__main__":
    main() 