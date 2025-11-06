import os
import random
from typing import List
import argparse
import run_einsum as einsum_experiments
import run_graph as graph_experiments
import plot as plot_experiments

BENCHMARK_REPEATS = int(os.environ.get('BENCHMARK_REPEATS', 5))
RESULT_DIR = os.environ.get("RESULT_DIR", "./results")


def figure7():
    #TODO: check if Figure 11 was already generated to reuse results
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
    pass

def figure9():
    pass

def figure10():
    print("[FIGURE 10]")
    result_path = f"{RESULT_DIR}/figure10"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    graph_experiments.run(result_path, graph_experiments.run_row_col_sparsity)
    result_file = f"{result_path}/bert_result.csv"
    plot_experiments.figure10(result_path, result_file)

def figure11():
    #TODO: reuse the .5 result from figure7
    pass


def figure12():
    print("[FIGURE 12]")
    result_path = f"{RESULT_DIR}/figure12"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    seed = random.randint(1, 1024)
    repeats = BENCHMARK_REPEATS
    einsum_experiments.run_for_sparsities(result_path, seed, repeats)
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

    for fig in figures:
        func = dispatch.get(fig, figure12)
        func()

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

    args = parser.parse_args()

    figures = [x.strip() for x in args.figures.split(",")]
    run(figures)
