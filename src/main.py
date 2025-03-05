from pipeline import (
    proposal_train,
    gt_train,
    gt_evaluate,
    rand_train,
    rand_evaluate,
    find_best_k_experiment,
    test_buffer,
)
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with arguments")
    parser.add_argument(
        "--exp",
        type=str,
        help="experiments(proposal/gt/rand/bm25/find_k/test_buffer/evaluate)",
    )
    parser.add_argument(
        "--method",  # buffer option
        type=str,
        help="buffer method(random_retrieve_reservoir_update/mir_retrieve_reservoir_update/gss_greedy_retrieve_reservoir_update/l2r_retrieve_l2r_update)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--use_label",  # proposal option
        action="store_true",
        help="when it is, use labeled positives. or use sampled positives.",
    )
    parser.add_argument(
        "--eval_cluster",  # proposal option
        action="store_true",
        help="when it is, evaluate model with clusters every each session.",
    )
    args = parser.parse_args()

    print(f"Running experiments: {args.exp}")
    if args.exp == "proposal":
        print(f"Use Label: {args.use_label}")
        print(f"Evaluate Cluster: {args.eval_cluster}")
        print(f"Number of Epochs: {args.num_epochs}")
        proposal_train(
            num_epochs=args.num_epochs,
            use_label=args.use_label,
            eval_cluster=args.eval_cluster,
        )
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
    elif args.exp == "test_buffer":
        method = args.method
        test_buffer(method)
    else:
        raise ValueError(f"Unsupported experiments {args.exp}")
