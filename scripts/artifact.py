import random
from typing import List
import argparse
import run_einsum as einsum_experiment



def figure7():
    seed = random.randint(1, 1024)
    sparsity = 0.5
    repeats = 5
    einsum_experiment.run(benchmark_dir, sparsity, seed, repeats)
    result_file = f"result{sparsity}{seed}{repeats}.txt"
    plot_figure7(result_file)

    pass

def run(figures: List[str]):
    pass

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
    )

    args = parser.parse_args()

    if args.figures:
        figures = [x.strip() for x in args.figures.split(",")]
    else:
        figures = ['7']
    run(figures)
