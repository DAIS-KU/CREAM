from pipeline import (
    proposal_train,
    gt_train,
    gt_evaluate,
    rand_train,
    rand_evaluate,
    find_best_k_experiment,
    test_buffer,
    er_train,
    er_evaluate,
    mir_train,
    mir_evaluate,
    ocs_train,
    ocs_evaluate,
    l2r_train,
    l2r_evaluate,
)
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with arguments")
    parser.add_argument(
        "--exp",
        type=str,
        help="experiments(proposal/gt/rand/bm25/find_k/test_buffer/er)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
    )
    # buffer option
    parser.add_argument(
        "--new_bz",
        default=3,
        help="sampling batch size of new documents.",
    )
    parser.add_argument(
        "--mem_upsample",
        default=6,
        help="candidates sampling from memory.",
    )
    parser.add_argument(
        "--mem_bz",
        default=3,
        help="sampling batch size of old documents from memory.",
    )
    parser.add_argument(
        "--comp",
        action="store_true",
        help="when it is, compatible",
    )
    # proposal option
    parser.add_argument(
        "--use_label",
        action="store_true",
        help="when it is, use labeled positives. or use sampled positives.",
    )
    parser.add_argument(
        "--eval_cluster",
        action="store_true",
        help="when it is, evaluate model with clusters every each session.",
    )
    parser.add_argument(
        "--negative_k",
        default=3,
        help="number of negative samples.",
    )
    args = parser.parse_args()

    print(f"Running experiments: {args.exp}")
    if args.exp == "proposal":
        print(f"Use Label: {args.use_label}")
        print(f"Evaluate Cluster: {args.eval_cluster}")
        print(f"Number of Epochs: {args.num_epochs}")
        proposal_train(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            use_label=args.use_label,
            negative_k=args.negative_k,
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
    elif args.exp == "er":
        er_train(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            compatible=args.comp,
            new_batch_size=args.new_bz,
            mem_batch_size=args.mem_bz,
        )
        er_evaluate()
    elif args.exp == "mir":
        mir_train(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            compatible=args.comp,
            new_batch_size=args.new_bz,
            mem_batch_size=args.mem_bz,
        )
        mir_evaluate()
    elif args.exp == "ocs":
        ocs_train(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            compatible=args.comp,
            new_batch_size=args.new_bz,
            mem_batch_size=args.mem_bz,
            mem_upsample=args.mem_upsample,
        )
        ocs_evaluate()
    elif args.exp == "l2r":
        l2r_train(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            compatible=args.comp,
            new_batch_size=args.new_bz,
            mem_batch_size=args.mem_bz,
            mem_upsample=args.mem_upsample,
        )
        l2r_evaluate()
    else:
        raise ValueError(f"Unsupported experiments {args.exp}")
