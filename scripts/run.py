#!/usr/bin/env python3

import subprocess
import re
import sys
from collections import defaultdict
import statistics

def run_cpp_program(program_path, sparsity, use_prop):
    """Run the C++ program with given parameters and return the output."""
    try:
        cmd = [program_path, str(sparsity), str(use_prop)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running program with sparsity={sparsity}, useProp={use_prop}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"Error: Program '{program_path}' not found")
        return None

def parse_output(output):
    """Parse the program output and extract metrics."""
    if not output:
        return None
    
    metrics = {}
    
    # Look for patterns like "metric = value"
    patterns = {
        'allocate': r'allocate\s*=\s*([\d.e+-]+)',
        'inference': r'inference\s*=\s*([\d.e+-]+)',
        'compile': r'compile\s*=\s*([\d.e+-]+)',
        'runtime': r'runtime\s*=\s*([\d.e+-]+)'
    }
    
    for metric, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[metric] = float(match.group(1))
        else:
            # Set inference to 0 if not present, others to None if missing
            metrics[metric] = 0.0 if metric == 'inference' else None
    
    return metrics

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <cpp_program_path>")
        print("Example: python script.py ./my_program")
        sys.exit(1)
    
    program_path = sys.argv[1]
    sparsity_values = [round(i * 0.1, 1) for i in range(10)]  # 0.0 to 0.9
    use_prop_values = [0, 1]
    num_runs = 1
    
    # Store all results: results[use_prop][sparsity][run_number] = metrics
    results = defaultdict(lambda: defaultdict(list))
    
    print("Running benchmarks...")
    print(f"Sparsity values: {sparsity_values}")
    print(f"UseProp values: {use_prop_values}")
    print(f"Number of runs per configuration: {num_runs}")
    print("-" * 60)
    
    # Run all combinations
    total_runs = len(sparsity_values) * len(use_prop_values) * num_runs
    current_run = 0
    
    for use_prop in use_prop_values:
        for sparsity in sparsity_values:
            print(f"\nRunning sparsity={sparsity}, useProp={use_prop}")
            
            for run in range(num_runs):
                current_run += 1
                print(f"  Run {run + 1}/{num_runs} ({current_run}/{total_runs})")
                
                output = run_cpp_program(program_path, sparsity, use_prop)
                metrics = parse_output(output)
                
                if metrics is None:
                    print(f"    Failed to parse output for run {run + 1}")
                    continue
                
                results[use_prop][sparsity].append(metrics)
                
                # Print current run results
                print(f"    allocate={metrics.get('allocate', 'N/A')}, "
                      f"inference={metrics.get('inference', 'N/A')}, "
                      f"compile={metrics.get('compile', 'N/A')}, "
                      f"runtime={metrics.get('runtime', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Process and display results for each useProp value
    for use_prop in use_prop_values:
        print(f"\n{'='*20} UseProp = {use_prop} {'='*20}")
        
        # Calculate averages and sums
        averages = {}
        sums = {}
        
        for sparsity in sparsity_values:
            if sparsity not in results[use_prop] or not results[use_prop][sparsity]:
                print(f"Warning: No valid results for sparsity={sparsity}, useProp={use_prop}")
                continue
            
            runs_data = results[use_prop][sparsity]
            metrics_by_type = defaultdict(list)
            
            # Group metrics by type across all runs
            for run_metrics in runs_data:
                for metric_name, value in run_metrics.items():
                    if value is not None:
                        metrics_by_type[metric_name].append(value)
            
            # Calculate averages and sums for this sparsity
            averages[sparsity] = {}
            sums[sparsity] = {}
            
            for metric_name, values in metrics_by_type.items():
                if values:
                    averages[sparsity][metric_name] = statistics.mean(values)
                    sums[sparsity][metric_name] = sum(values)
                else:
                    averages[sparsity][metric_name] = 0.0
                    sums[sparsity][metric_name] = 0.0
        
        # Display averages
        print(f"\nAVERAGES (over {num_runs} runs):")
        print(f"{'Sparsity':<10} {'Allocate':<12} {'Inference':<12} {'Compile':<12} {'Runtime':<12}")
        print("-" * 70)
        
        for sparsity in sparsity_values:
            if sparsity in averages:
                avg = averages[sparsity]
                print(f"{sparsity:<10.1f} "
                      f"{avg.get('allocate', 0):<12.6f} "
                      f"{avg.get('inference', 0):<12.6f} "
                      f"{avg.get('compile', 0):<12.6f} "
                      f"{avg.get('runtime', 0):<12.6f}")
        
        # Display sums
        print(f"\nSUMS (over {num_runs} runs):")
        print(f"{'Sparsity':<10} {'Allocate':<12} {'Inference':<12} {'Compile':<12} {'Runtime':<12}")
        print("-" * 70)
        
        for sparsity in sparsity_values:
            if sparsity in sums:
                sum_vals = sums[sparsity]
                print(f"{sparsity:<10.1f} "
                      f"{sum_vals.get('allocate', 0):<12.6f} "
                      f"{sum_vals.get('inference', 0):<12.6f} "
                      f"{sum_vals.get('compile', 0):<12.6f} "
                      f"{sum_vals.get('runtime', 0):<12.6f}")
    
    print(f"\n{'='*80}")
    print("Benchmarking complete!")

if __name__ == "__main__":
    main()
