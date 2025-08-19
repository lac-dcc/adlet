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

def run(benchmark_dir: str, sparsity: float, seed: int, n: int):
    files = os.listdir(benchmark_dir)
    errors = []
    with open(f"result{sparsity}{seed}{n}.txt", "wt") as result_file:
        result_file.write('file_name,format,sparsity,propagate,ratio_before,ratio_after,analysis,load_time,compilation_time,runtime, overall_memory, tensors-size\n')
        for idx, file in enumerate(files):
            file_path = f"{benchmark_dir}{file}"
            for format_str in ["sparse", "dense"]:
                for propagate in [0, 1]:
                    if format_str == "dense" and propagate == 1:
                        continue
                    cmd = ["./benchmark", "einsum", file_path, format_str, str(sparsity), str(propagate), str(seed)]
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
                        result_line = f'{file},{format_str},{sparsity}, {propagate}, {mean_metrics["before"]}, {mean_metrics["after"]}, {mean_metrics["analysis"]}, {mean_metrics["load"]}, {mean_metrics["compilation"]}, {mean_metrics["runtime"]}, {mean_metrics["memory"]}, {mean_metrics["tensors-size"]}'
                        result_file.write(result_line + "\n")
                    except Exception as e:
                        errors.append(file)
                        print(f"Error running {file_path}: {str(e)}")

    with open("errors.txt", "wt") as error_file:
        error_file.write("\n".join(errors))

def run_for_sparsties(benchmark_dir: str, seed: int, n: int):
    sparsities = [0.9, 0.7, 0.5, 0.3]
    for sparsity in sparsities:
        run(benchmark_dir, sparsity, seed, n)

def run_with_timeout(benchmark_dir:str, sparsity: float, seed: int, timeout_seconds: int):
    print("Testing benchmarks...")
    files = os.listdir(benchmark_dir)
    propagate = 0
    with open("filter.txt", "wt") as result_file:
        for idx, file in enumerate(files):
            print(f"{idx}/{len(files)}", end="\r")
            file_path = f"{benchmark_dir}{file}"
            cmd = ["./benchmark", "einsum", file_path, "dense", str(sparsity), str(propagate), str(seed)]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
                metrics = parse_output(result.stdout)
                _ = metrics["runtime"]
                print("success!")
                result_file.write(f"success {file}\n")
            except subprocess.TimeoutExpired:
                print("timeout!")
                result_file.write(f"timeout {file}\n")
            except Exception as e:
                print(f"error {str(e)}")
                result_file.write(f"error {file}\n")
            result_file.flush()


if __name__ == "__main__":
    benchmark_dir = sys.argv[1]
    sparsity = float(sys.argv[2])
    seed = int(sys.argv[3])
    repeats = int(sys.argv[4])
    #run(benchmark_dir, sparsity, seed, repeats)
    #run_with_timeout(benchmark_dir, sparsity, seed, 900)
    run_for_sparsties(benchmark_dir, seed, repeats)

