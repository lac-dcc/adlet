import os
import statistics
import subprocess

BUILD_DIR = os.environ.get("BUILD_PATH", "./build")
BIN_PATH = os.environ.get("BIN_PATH", "./build/benchmark")
SPA_ROOT = os.environ.get("SPA_ROOT", ".")
TESA_BIN_PATH = os.environ.get("BIN_PATH", "./build/tesa-prop")
BENCHMARK_REPEATS = int(os.environ.get('BENCHMARK_REPEATS', 5))

def parse_output(output):
    metrics = {}
    lines = output.strip().splitlines()
    for line in lines:
        if "proptime" in line:
            metrics["proptime"] = float(line.split("=")[-1].strip())
    return metrics

def recompile_size(root_dir: str, build_dir: str, size: int):
    cmd = ["cmake", "-S" + root_dir, "-B" + build_dir, "-G", "Ninja", f"-DSIZE_MACRO={size}"]
    process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    process.wait()
    cmd = ["cmake", "--build", build_dir]
    process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    process.wait()

def run_spa(result_dir: str, size: int, n: int):
    recompile_size(SPA_ROOT, BUILD_DIR, size)
    errors = []
    with open(f"{result_dir}/proptime_spa_result_{size}.csv", "wt") as result_file:
        result_file.write('size,proptime\n')
        times = {"proptime":[]}
        print(f"[running proptime for SPA size {size}]")
        try:
            for i in range(n):
                cmd = [BIN_PATH, "proptime"]
                print(f"iteration {i + 1}/{n}",  end="\r")
                process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                process.wait()
                metrics = parse_output(process.stdout.read())
                for k in times:
                    times[k].append(metrics.get(k, 0.0))

            mean_metrics = {k: statistics.mean(times[k]) for k in times}
            result_line = f'{size},{mean_metrics["proptime"]}'
            result_file.write(result_line + "\n")
            result_file.flush()
        except Exception as e:
            print(f"Error: {str(e)}")

    with open("errors.txt", "wt") as error_file:
        error_file.write("\n".join(errors))

def run_tesa(result_dir: str, size: int, n: int):
    recompile_size(SPA_ROOT, BUILD_DIR, size)
    errors = []
    with open(f"{result_dir}/proptime_tesa_result_{size}.csv", "wt") as result_file:
        result_file.write('size,proptime\n')
        times = {"proptime":[]}
        print(f"[running proptime for TeSA size {size}]")
        try:
            for i in range(n):
                cmd = [TESA_BIN_PATH, "proptime"]
                print(f"iteration {i + 1}/{n}",  end="\r")
                process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                process.wait()
                metrics = parse_output(process.stdout.read())
                for k in times:
                    times[k].append(metrics.get(k, 0.0))

            mean_metrics = {k: statistics.mean(times[k]) for k in times}
            result_line = f'{size},{mean_metrics["proptime"]}'
            result_file.write(result_line + "\n")
            result_file.flush()
        except Exception as e:
            print(f"Error: {str(e)}")

    with open("errors.txt", "wt") as error_file:
        error_file.write("\n".join(errors))

def run_prop(result_dir: str, sparsity: float, seed: int, n: int):
    files = os.listdir(EINSUM_DATASET)
    errors = []
    run_fw = 1
    with open(f"{result_dir}/einsum_result_prop_{sparsity}_{seed}_{n}.csv", "wt") as result_file:
        result_file.write('file_name,sparsity,run_fw,run_lat,run_bw,initial_ratio,fw_ratio,lat_ratio,bw_ratio\n')
        for idx, file in enumerate(files):
            file_path = f"{EINSUM_DATASET}{file}"
            for run_lat in [0, 1]:
                for run_bw in [0, 1]:
                    data = {"fw_ratio":[], "lat_ratio": [], "bw_ratio": [], "initial_ratio": []}
                    print(f"[running {idx}/{len(files)}]: {file} - run_fw={run_fw}, run_lat={run_lat}, run_bw={run_bw}")
                    try:
                        for i in range(n):
                            cmd = [BIN_PATH, "einsum", "prop", file_path, str(sparsity), str(run_fw), str(run_lat), str(run_bw), str(seed + i)]
                            print(f"iteration {i}/{n}",  end="\r")
                            process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                            process.wait()
                            metrics = parse_output(process.stdout.read())
                            for k in data:
                                data[k].append(metrics.get(k, 0.0))

                        mean_metrics = {k: statistics.mean(data[k]) for k in data}
                        result_line = f'{file},{sparsity},{run_fw},{run_lat},{run_bw},{mean_metrics["initial_ratio"]},{mean_metrics["fw_ratio"]},{mean_metrics["lat_ratio"]},{mean_metrics["bw_ratio"]}'
                        result_file.write(result_line + "\n")
                        result_file.flush()
                    except Exception as e:
                        errors.append(file)
                        print(f"Error running {file_path}: {str(e)}")

    with open("errors.txt", "wt") as error_file:
        error_file.write("\n".join(errors))

def run_for_sparsities(result_dir: str, seed: int, n: int):
    sparsities = [0.9, 0.7, 0.5, 0.3]
    for sparsity in sparsities:
        run(result_dir, sparsity, seed, n)

def run_prop_for_sparsities(result_dir: str, seed: int, n: int):
    sparsities = [0.9, 0.7, 0.5, 0.3]
    for sparsity in sparsities:
        run_prop(result_dir, sparsity, seed, n)

def run_with_timeout(result_dir: str, sparsity: float, seed: int, timeout_seconds: int):
    print("Testing benchmarks...")
    files = os.listdir(EINSUM_DATASET)
    propagate = 0
    with open("filter.txt", "wt") as result_file:
        for idx, file in enumerate(files):
            print(f"{idx}/{len(files)}", end="\r")
            file_path = f"{EINSUM_DATASET}{file}"
            cmd = [BIN_PATH, "einsum", file_path, "dense", str(sparsity), str(propagate), str(seed)]
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
    import random
    seed = random.randint(1, 1024)
    # run_for_sparsities("./", seed, BENCHMARK_REPEATS)
    run("./", 0.5, 12, 1)


