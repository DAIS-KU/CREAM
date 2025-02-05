from pipeline import (
    proposal_train,
    proposal_evaluate,
    gt_train,
    gt_evaluate,
    rand_train,
    rand_evaluate,
)
from cluster import find_best_k_experiment
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with arguments")
    parser.add_argument("--exp", type=str, help="experiments(proposal/gt/rand/find_k)")
    args = parser.parse_args()

    print(f"Running experiments: {args.exp}")
    if args.exp == "proposal":
        proposal_train()
        proposal_evaluate()
    elif args.exp == "gt":
        gt_train()
        gt_evaluate()
    elif args.exp == "rand":
        rand_train()
        rand_evaluate()
    elif ards.exp == "find_k":
        find_best_k_experiment(start=30, end=60, gap=10, max_iter=5)
    else:
        raise ValueError(f"Unsupported experiments {args.exp}")
