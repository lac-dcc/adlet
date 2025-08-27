import random
import statistics
from datetime import datetime
import subprocess

#only column sparsity
run1 = [["0.0", "0.1", "SparseDense", "0"], ["0.0", "0.3", "SparseDense", "0"], ["0.0", "0.5", "SparseDense", "0"], ["0.0", "0.7", "SparseDense", "0"], ["0.0", "0.9", "SparseDense", "0"], ["0.0", "0.1", "SparseDense", "1"], ["0.0", "0.3", "SparseDense", "1"], ["0.0", "0.5", "SparseDense", "1"], ["0.0", "0.7", "SparseDense", "1"], ["0.0", "0.9", "SparseDense", "1"], ["0.0", "0.1", "DD", "0"], ["0.0", "0.3", "DD", "0"], ["0.0", "0.5", "DD", "0"], ["0.0", "0.7", "DD", "0"], ["0.0", "0.9", "DD", "0"]]

#only row sparsity
run2 = [["0.1", "0.0", "SparseDense", "0"], ["0.3", "0.0", "SparseDense", "0"], ["0.5", "0.0", "SparseDense", "0"], ["0.7", "0.0", "SparseDense", "0"], ["0.9", "0.0", "SparseDense", "0"], ["0.1", "0.0", "SparseDense", "1"], ["0.3", "0.0", "SparseDense", "1"], ["0.5", "0.0", "SparseDense", "1"], ["0.7", "0.0", "SparseDense", "1"], ["0.9", "0.0", "SparseDense", "1"], ["0.1", "0.0", "DD", "0"], ["0.3", "0.0", "DD", "0"], ["0.5", "0.0", "DD", "0"], ["0.7", "0.0", "DD", "0"], ["0.9", "0.0", "DD", "0"]]

#row and column sparsity
run3 = [["0.1", "0.1", "SparseDense", "0"], ["0.3", "0.3", "SparseDense", "0"], ["0.5", "0.5", "SparseDense", "0"], ["0.7", "0.7", "SparseDense", "0"], ["0.9", "0.9", "SparseDense", "0"], ["0.1", "0.1", "SparseDense", "1"], ["0.3", "0.3", "SparseDense", "1"], ["0.5", "0.5", "SparseDense", "1"], ["0.7", "0.7", "SparseDense", "1"], ["0.9", "0.9", "SparseDense", "1"], ["0.1", "0.1", "DD", "0"], ["0.3", "0.3", "DD", "0"], ["0.5", "0.5", "DD", "0"], ["0.7", "0.7", "DD", "0"], ["0.9", "0.9", "DD", "0"]]

#all
run4 = [["0.0", "0.1", "SparseDense", "0"], ["0.0", "0.3", "SparseDense", "0"], ["0.0", "0.5", "SparseDense", "0"], ["0.0", "0.7", "SparseDense", "0"], ["0.0", "0.9", "SparseDense", "0"], ["0.0", "0.1", "SparseDense", "1"], ["0.0", "0.3", "SparseDense", "1"], ["0.0", "0.5", "SparseDense", "1"], ["0.0", "0.7", "SparseDense", "1"], ["0.0", "0.9", "SparseDense", "1"], ["0.0", "0.1", "DD", "0"], ["0.0", "0.3", "DD", "0"], ["0.0", "0.5", "DD", "0"], ["0.0", "0.7", "DD", "0"], ["0.0", "0.9", "DD", "0"], ["0.1", "0.0", "SparseDense", "0"], ["0.3", "0.0", "SparseDense", "0"], ["0.5", "0.0", "SparseDense", "0"], ["0.7", "0.0", "SparseDense", "0"], ["0.9", "0.0", "SparseDense", "0"], ["0.1", "0.0", "SparseDense", "1"], ["0.3", "0.0", "SparseDense", "1"], ["0.5", "0.0", "SparseDense", "1"], ["0.7", "0.0", "SparseDense", "1"], ["0.9", "0.0", "SparseDense", "1"], ["0.1", "0.0", "DD", "0"], ["0.3", "0.0", "DD", "0"], ["0.5", "0.0", "DD", "0"], ["0.7", "0.0", "DD", "0"], ["0.9", "0.0", "DD", "0"], ["0.1", "0.1", "SparseDense", "0"], ["0.3", "0.3", "SparseDense", "0"], ["0.5", "0.5", "SparseDense", "0"], ["0.7", "0.7", "SparseDense", "0"], ["0.9", "0.9", "SparseDense", "0"], ["0.1", "0.1", "SparseDense", "1"], ["0.3", "0.3", "SparseDense", "1"], ["0.5", "0.5", "SparseDense", "1"], ["0.7", "0.7", "SparseDense", "1"], ["0.9", "0.9", "SparseDense", "1"], ["0.1", "0.1", "DD", "0"], ["0.3", "0.3", "DD", "0"], ["0.5", "0.5", "DD", "0"], ["0.7", "0.7", "DD", "0"], ["0.9", "0.9", "DD", "0"]]

seed = "1"
repeats = 5
binary = "./benchmark"
graph_name = "bert"

def parse_output(output):
    metrics = {}
    lines = output.strip().splitlines()
    for line in lines:
        if "analysis" in line:
            metrics["analysis"] = float(line.split("=")[-1].strip())
        elif "load graph" in line:
            metrics["load"] = float(line.split("=")[-1].strip())
        elif "compilation" in line:
            metrics["compilation"] = float(line.split("=")[-1].strip())
        elif "runtime" in line:
            metrics["runtime"] = float(line.split("=")[-1].strip())
        elif "memory used" in line:
            metrics["memory"] = float(line.split("=")[-1].strip())
        elif "before" in line:
            metrics["before"] = float(line.split("=")[-1].strip())
        elif "after" in line:
            metrics["after"] = float(line.split("=")[-1].strip())
    return metrics

def run(list_runs, warmup=False):
    results = []
    total = len(list_runs) * repeats
    iter_count = 0
    file_name = f"{graph_name}-{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}-random-{str(warmup).lower()}.out"
    with open(file_name, "wt") as file:
        file.write("model, row, col, format, prop, before, after, analysis, load, comp, run, memory\n")
        for params in list_runs:
            left_sparsity, right_sparsity = params[0], params[1]
            fmt = params[2]
            opt = params[3]
            config_name = f"{left_sparsity},{right_sparsity},{fmt},opt={opt}"
            times = {"before":[], "after": [], "analysis": [], "load": [], "compilation": [], "runtime": [], "memory": []}
            cmd = [binary, "graph", graph_name, left_sparsity, right_sparsity, fmt, opt, seed]

            if warmup:
                result = subprocess.run(cmd, capture_output=True, text=True)
                metrics = parse_output(result.stdout)

            print(f"{iter_count}/{total} - {config_name}", end='\r')

            for _ in range(repeats):
                result = subprocess.run(cmd, capture_output=True, text=True)
                iter_count+=1
                metrics = parse_output(result.stdout)
                for k in times:
                    times[k].append(metrics.get(k, 0.0))
        
            mean_metrics = {k: statistics.mean(times[k]) for k in times}
            mean_metrics["config"] = config_name
            result_line = f'{graph_name},{left_sparsity}, {right_sparsity}, {fmt}, {opt}, {mean_metrics["before"]}, {mean_metrics["after"]}, {mean_metrics["analysis"]}, {mean_metrics["load"]}, {mean_metrics["compilation"]}, {mean_metrics["runtime"]}, {mean_metrics["memory"]}'
            
            file.write(f'{result_line}\n')
            results.append(mean_metrics)
    return results

def random_run(list_runs):
    config = {}
    for run in list_runs:
        config[",".join(run)] = [run, repeats]
    
    def get_params():
        size = len(config)
        while size:
            pick = random.randint(0, size - 1)
            items_list = list(config)
            key = items_list[pick]
            yield key, config[key][0]
            if config[key][1] == 1:
                del config[key]
                size -= 1
            else:
                config[key][1] -= 1
    results = []
    times = {}
    total = len(list_runs) * repeats
    iter_count = 0
    file_name = f"{graph_name}-{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}-random.out"
    with open(file_name, "wt") as file:
        file.write("model, row, col, format, prop, before, after, analysis, load, comp, run, memory\n")
        for name, params in get_params():
            left_sparsity, right_sparsity, fmt, opt = params[0], params[1], params[2], params[3]
            cmd = [binary, "graph", graph_name, left_sparsity, right_sparsity, fmt, opt]
            result = subprocess.run(cmd, capture_output=True, text=True)
            iter_count+=1
            metrics = parse_output(result.stdout)
            if name not in times:
                times[name] = {}
            for k, v in metrics.items():
                if k in times[name]:
                    times[name][k].append(v)
                else:
                    times[name][k] = [v]
            print(f"{iter_count}/{total} - {name}", end='\r')

        for config_name, values in times.items():
            mean_metrics = {k: statistics.mean(values[k]) for k in values}
            mean_metrics["config"] = config_name
            left_sparsity, right_sparsity, fmt, opt = config_name.split(",")
            result_line = f'{graph_name},{left_sparsity}, {right_sparsity}, {fmt}, {opt}, {mean_metrics["before"]}, {mean_metrics["after"]}, {mean_metrics["analysis"]}, {mean_metrics["load"]}, {mean_metrics["compilation"]}, {mean_metrics["runtime"]}, {mean_metrics["memory"]}'
            file.write(f'{result_line}\n')
            results.append(mean_metrics)
        return results

def run_all():
    run(run1, False)
    run(run2, False)
    run(run3, False)


if __name__ == "__main__":
    run(list_runs=runs, warmup=False)

