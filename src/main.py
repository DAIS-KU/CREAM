from pipeline import (
    proposal_train,
    proposal_evaluate,
    gt_train,
    gt_evaluate,
    rand_train,
    rand_evaluate,
    bm25_evaluate,
    find_best_k_experiment,
)
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with arguments")
    parser.add_argument(
        "--exp", type=str, help="experiments(proposal/gt/rand/bm25/find_k)"
    )
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
    elif args.exp == "bm25":
        bm25_evaluate()
    elif args.exp == "find_k":
        find_best_k_experiment(start=200, end=600, gap=100, max_iters=5)
    else:
        raise ValueError(f"Unsupported experiments {args.exp}")
