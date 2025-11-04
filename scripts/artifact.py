import os
import random
from typing import List
import argparse
import run_einsum as einsum_experiments
import plot as plot_experiments

EINSUM_DATASET = os.environ.get('EINSUM_DATASET', 'einsum-dataset')
BENCHMARK_REPEATS = int(os.environ.get('BENCHMARK_REPEATS', 1))


def figure7():
    print("[FIGURE 7]")
    seed = random.randint(1, 1024)
    sparsity = 0.5
    repeats = BENCHMARK_REPEATS
    einsum_experiments.run(EINSUM_DATASET, sparsity, seed, repeats)
    result_file = f"result_{sparsity}_{seed}_{repeats}.csv"
    plot_experiments.figure7(result_file)



def run(figures: List[str]):
    for fig in figures:
        if fig == '7':
            figure7()


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
