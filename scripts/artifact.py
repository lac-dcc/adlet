import os
import random
from typing import List
import argparse
import run_einsum as einsum_experiments
import run_proptime as proptime_experiments
import run_graph as graph_experiments
import plot as plot_experiments

BENCHMARK_REPEATS = 5 
RESULT_DIR = os.environ.get("RESULT_DIR", "./results")


def figure7():
    print("[FIGURE 7]")
    result_path = f"{RESULT_DIR}/figure7"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    seed = random.randint(1, 1024)
    sparsity = 0.5
    repeats = BENCHMARK_REPEATS
    einsum_experiments.run(result_path, sparsity, seed, repeats)
    result_file = f"{result_path}/einsum_result_{sparsity}_{seed}_{repeats}.csv"
    plot_experiments.figure7(result_path, result_file)

def figure8():
    print("[FIGURE 8]")
    result_path = f"{RESULT_DIR}/figure8"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    repeats = 10 # very high variance so set higher

    sizes = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]
    for size in sizes:
        proptime_experiments.run_spa(result_path, size, repeats)
        result_file = f"{RESULT_DIR}/proptime_spa_result_{size}.csv"
        proptime_experiments.run_tesa(result_path, size, repeats)
        result_file = f"{RESULT_DIR}/proptime_tesa_result_{size}.csv"
    proptime_experiments.recompile_spa_size(".", "./build", 2048)
    plot_experiments.figure8(result_path)

def figure9():
    print("[FIGURE 9]")
    result_path = f"{RESULT_DIR}/figure9"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    sparsity = 0.5
    repeats = BENCHMARK_REPEATS
    sparsities = [0.9, 0.7, 0.5, 0.3]
    seed = random.randint(1, 1024) # probably should generate a fixed set of seeds for final artifact
    for sparsity in sparsities:
        einsum_experiments.run_prop(result_path, sparsity, seed, repeats)
        result_file = f"{result_path}/einsum_result_{sparsity}_{seed}_{repeats}.csv"
    plot_experiments.figure9(result_path)

def figure10():
    print("[FIGURE 10]")
    result_path = f"{RESULT_DIR}/figure10"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    graph_experiments.run(result_path, graph_experiments.run_row_col_sparsity, repeat=BENCHMARK_REPEATS)
    result_file = f"{result_path}/bert_result.csv"
    plot_experiments.figure10(result_path, result_file)

def figure11():
    print("[FIGURE 11]")
    result_path = f"{RESULT_DIR}/figure11"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    seed = random.randint(1, 1024) # probably should generate a fixed set of seeds for final artifact
    repeats = BENCHMARK_REPEATS
    sparsities = [0.9, 0.7, 0.5, 0.3, 0.1]
    for sparsity in sparsities:
        einsum_experiments.run_benchmark_12(result_path, sparsity, seed, repeats)
    plot_experiments.figure11(result_path)

def figure12():
    print("[FIGURE 12]")
    result_path = f"{RESULT_DIR}/figure12"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    seed = random.randint(1, 1024)
    repeats = BENCHMARK_REPEATS
    einsum_experiments.run_for_sparsities(result_path, seed, repeats, compute=0)
    plot_experiments.figure12(result_path)

def run(figures: List[str]):
    dispatch = {
        '7': figure7,
        '8': figure8,
        '9': figure9,
        '10': figure10,
        '11': figure11,
        '12': figure12,
    }

    print(f"Building figures {figures}")
    for fig in figures:
        func = dispatch.get(fig, figure12)
        try:
            func()
        except Exception as e:
            print(f"Error  running figure {fig} - {str(e)}")

if __name__ == "__main__":
    # Figure 7 - SPA runtime vs compilation vs execution time (einsum)
    # Figure 8 - SPA vs SparTA
    # Figure 9 - Sparsity propagation comparison FLB (einsum)
    # Figure 10 - BERT-like runtime comparison
    # Figure 11 - Einsum-12 runtime comparison
    # Figure 12 - Memory reduction (einsum)
    #artifact.py --figures 7, 8, 9
    #artifact.py --figures # slow

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figures",
        type=str,
        help="Comma-separated list of figures",
        default="7,8,9,10,11,12",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        help="Number of executions for each benchmark",
        default=5,
    )

    args = parser.parse_args()

    figures = [x.strip() for x in args.figures.split(",")]
    BENCHMARK_REPEATS = args.repeat
    run(figures)
