import statistics
import sys
import os
import subprocess


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
        elif "tensors" in line:
            metrics["tensors-size"] = float(line.split("=")[-1].strip())
    return metrics

def run(benchmark_dir: str, sparsity: float, prob_to_prune: float, seed: int, n: int):
    files = os.listdir(benchmark_dir)
    errors = []
    with open(f"result{sparsity}{prob_to_prune}{seed}{n}.txt", "wt") as result_file:
        result_file.write('file_name,format,sparsity,prob_to_prune,propagate,ratio_before,after,analysis,load_time,compilation_time,run_time, overall_memory, tensors-size\n')
        for idx, file in enumerate(files):
            file_path = f"{benchmark_dir}{file}"
            for format_str in ["sparse", "dense"]:
                for propagate in [0, 1]:
                    if format_str == "dense" and propagate == 1:
                        continue
                    cmd = ["./benchmark", "einsum", file_path, format_str, str(sparsity), str(prob_to_prune), str(propagate), str(seed)]
                    times = {"before":[], "after": [], "analysis": [], "load": [], "compilation": [], "runtime": [], "memory": [], "tensors-size":[]}
                    print(f"[running {idx}/{len(files)}]: {file} - format={format_str} - prop={propagate}")
                    try:
                        for i in range(n):
                            print(f"iteration {i}/{n}",  end="\r")
                            process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                            process.wait()
                            metrics = parse_output(process.stdout.read())
                            for k in times:
                                times[k].append(metrics.get(k, 0.0))

                        mean_metrics = {k: statistics.mean(times[k]) for k in times}
                        result_line = f'{file},{format_str},{sparsity}, {prob_to_prune}, {propagate}, {mean_metrics["before"]}, {mean_metrics["after"]}, {mean_metrics["analysis"]}, {mean_metrics["load"]}, {mean_metrics["compilation"]}, {mean_metrics["runtime"]}, {mean_metrics["memory"]}, {mean_metrics["tensors-size"]}'
                        result_file.write(result_line + "\n")
                    except Exception as e:
                        errors.append(file)
                        print(f"Error running {file_path}: {str(e)}")

    with open("errors.txt", "wt") as error_file:
        error_file.write("\n".join(errors))

def run_with_timeout(benchmark_dir:str, sparsity: float, prob_to_prune:float, seed: int, timeout_seconds: int):
    print("Testing benchmarks...")
    files = os.listdir(benchmark_dir)
    timed_out = []
    errors = []
    below_timeout = []
    propagate = 0
    for idx, file in enumerate(files):
        print(f"[running]: {file}")
        print(f"{idx}/{len(files)}", end="\r")
        file_path = f"{benchmark_dir}{file}"
        cmd = ["./benchmark", "einsum", file_path, "dense", str(sparsity), str(prob_to_prune), str(propagate), str(seed)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
            metrics = parse_output(result.stdout)
            below_timeout.append(file)
        except subprocess.TimeoutExpired:
            print(f"{file} timeout!")
            timed_out.append(file)
        except Exception as e:
            print(f"{file} error: {str(e)}")
            errors.append(file)

    with open("time_out.txt", "wt") as t_file:
        t_file.write("\n".join(timed_out))

    with open("errors.txt", "wt") as e_file:
        e_file.write("\n".join(errors))

    with open("working.txt", "wt") as b_file:
        b_file.write("\n".join(below_timeout))

if __name__ == "__main__":
    benchmark_dir = sys.argv[1]
    sparsity = float(sys.argv[2])
    prob_to_prune = float(sys.argv[3])
    seed = int(sys.argv[4])
    repeats = int(sys.argv[5])
    run(benchmark_dir, sparsity, prob_to_prune, seed, repeats)
    # run_with_timeout(benchmark_dir, sparsity, prob_to_prune, seed, 60)

