import sys
import os
import subprocess


SPARSITY = 0.5
PROB_TO_PRUNE = 0.5
SEED = 4


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

def run(benchmark_dir: str):
    files = os.listdir(benchmark_dir)
    with open("result.txt", "wt") as result_file:
        result_file.write('file_name,sparsity,prob_to_prune,propagate,ratio_before,after,analysis,load_time,compilation_time,run_time, overall_memory, tensors-size\n')

        for file in files:
            print(f"running {file}")
            file_path = f"./sub100/{file}"
            for propagate in [0, 1]:
                cmd = ["./benchmark", "einsum", file_path, str(SPARSITY), str(PROB_TO_PRUNE), str(propagate), str(SEED)]
                process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                process.wait()
                mean_metrics = parse_output(process.stdout)

                result_line = f'{file},{SPARSITY}, {PROB_TO_PRUNE}, {propagate}, {mean_metrics["before"]}, {mean_metrics["after"]}, {mean_metrics["analysis"]}, {mean_metrics["load"]}, {mean_metrics["compilation"]}, {mean_metrics["runtime"]}, {mean_metrics["memory"]}, {mean_metrics["tensors-size"]}'
                result_file.write(result_line + "\n")



if __name__ == "__main__":
    benchmark_dir = sys.argv[1]
    run(benchmark_dir)
