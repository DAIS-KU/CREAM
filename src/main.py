import argparse
import glob
import json
import os

from pipeline import (
    evaluate_wo_cluster,
    evaluate_wo_term,
    inc_evalutate,
    inc_train,
    l2r_evaluate,
    l2r_train,
    proposal_ance_evaluate,
    proposal_ance_train,
    proposal_evaluate,
    proposal_rerank_evaluate,
    proposal_rerank_train,
    proposal_train,
    train_wo_cluster,
    train_wo_term,
    waringup_evaluate,
    waringup_train,
)


def validate_json_files(file_paths):
    for file_path in file_paths:
        print(f"file_path: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        print(
                            f"❌ JSONDecodeError in {file_path} at line {line_num}: {e}"
                        )
                        print(f"👉 Problematic line: {line.strip()}")
                        break
                else:
                    print(f"✅ Valid JSON: {file_path}")
        except Exception as e:
            print(f"❌ Error opening file {file_path}: {e}")


def validate_data():
    base_path = "/home/work/retrieval/data/sub"
    file_patterns = [
        "train_session*_docs.jsonl",
        "train_session*_queries.jsonl",
        "test_session*_docs.jsonl",
        "test_session*_queries.jsonl",
    ]
    file_list = []
    for pattern in file_patterns:
        matched_files = glob.glob(f"{base_path}/{pattern}")
        print(f"🔍 Found {len(matched_files)} files for pattern: {pattern}")
        file_list.extend(matched_files)
    validate_json_files(file_list)


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
        "--use_weight",
        action="store_true",
        help="apply time-decayed weight.",
    )
    parser.add_argument(
        "--use_tensor_key",
        action="store_true",
        help="Use hash as tensor instead of map.",
    )
    parser.add_argument(
        "--load_cluster",
        action="store_true",
        help="start with loading previous session cluster",
    )
    parser.add_argument(
        "--warming_up_method",
        default=None,
        type=str,
        help="use cluster warming up for the first session.(doc only/query only/mixed)",
    )
    parser.add_argument(
        "--start",
        default=0,
        type=int,
        help="start session number.",
    )
    parser.add_argument(
        "--end",
        default=4,
        type=int,
        help="end session number.",
    )
    parser.add_argument(
        "--negative_k",
        default=6,
        type=int,
        help="number of negative samples.",
    )
    parser.add_argument(
        "--nbits",
        default=12,
        type=int,
        help="number of hash bits.",
    )
    parser.add_argument(
        "--mi",
        default=3,
        type=int,
        help="max iters to converge centroids.",
    )
    parser.add_argument(
        "--wr",
        default=None,
        type=float,
        help="warming up rate",
    )
    parser.add_argument(
        "--init_k",
        type=int,
        default=None,
        help="warming up k cluster",
    )
    parser.add_argument(
        "--cmnsz",
        type=int,
        default=10,
        help="cluster minimum number of instances before adding new centroids.",
    )
    parser.add_argument(
        "--sr",
        type=float,
        default=None,
        help="Document stream sampling rate",
    )
    parser.add_argument(
        "--sspq",
        type=int,
        default=None,
        help="Document stream candidates size for each query",
    )
    parser.add_argument(
        "--rdsz",
        type=int,
        default=None,
        help="Valid cluster document size requirenebts",
    )
    args = parser.parse_args()

    print(f"Running experiments: {args.exp}")
    if args.exp == "proposal":
        print(f"Start session: {args.start}")
        print(f"End session: {args.end}")
        print(f"Load cluster: {args.load_cluster}")
        print(f"Use cluster warming up: {args.warming_up_method}")
        print(f"Use label: {args.use_label}")
        print(f"Use weight: {args.use_weight}")
        print(f"Use tensor key: {args.use_tensor_key}")
        print(f"Use nbits: {args.nbits}")
        print(f"Use max_iters: {args.mi}")
        print(f"Use warmingup_rate: {args.wr}")
        print(f"Use warmingup_k: {args.init_k}")
        print(f"Use cluster_min_size: {args.cmnsz}")
        print(f"Use required_doc_size: {args.rdsz}")
        print(f"Use stream sampling rate: {args.sr}")
        print(f"Use sampling size per query: {args.sspq}")
        print(f"Number of Epochs: {args.num_epochs}")
        proposal_train(
            start_session_number=args.start,
            end_sesison_number=args.end,
            load_cluster=args.load_cluster,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            negative_k=args.negative_k,
            use_label=args.use_label,
            use_weight=args.use_weight,
            use_tensor_key=args.use_tensor_key,
            max_iters=args.mi,
            warmingup_rate=args.wr,
            init_k=args.init_k,
            cluster_min_size=args.cmnsz,
            sampling_rate=args.sr,
            sampling_size_per_query=args.sspq,
            nbits=args.nbits,
            warming_up_method=args.warming_up_method,
            required_doc_size=args.rdsz,
        )
        # proposal_evaluate(0)
    elif args.exp == "wu":
        waringup_train()
        waringup_evaluate()
    elif args.exp == "val":
        validate_data()
    elif args.exp == "l2r":
        l2r_train()
        l2r_evaluate()
    elif args.exp == "inc":
        inc_train()
        inc_evalutate()
    elif args.exp == "ablation_cluster":
        train_wo_cluster()
        evaluate_wo_cluster()
    elif args.exp == "ablation_term":
        train_wo_term()
        evaluate_wo_term()
    elif args.exp == "proposal_rerank":
        print(f"Start session: {args.start}")
        print(f"End session: {args.end}")
        print(f"Load cluster: {args.load_cluster}")
        print(f"Use cluster warming up: {args.warming_up_method}")
        print(f"Use label: {args.use_label}")
        print(f"Use weight: {args.use_weight}")
        print(f"Use tensor key: {args.use_tensor_key}")
        print(f"Use nbits: {args.nbits}")
        print(f"Use max_iters: {args.mi}")
        print(f"Use warmingup_rate: {args.wr}")
        print(f"Use warmingup_k: {args.init_k}")
        print(f"Use cluster_min_size: {args.cmnsz}")
        print(f"Use required_doc_size: {args.rdsz}")
        print(f"Use stream sampling rate: {args.sr}")
        print(f"Use sampling size per query: {args.sspq}")
        print(f"Number of Epochs: {args.num_epochs}")
        proposal_rerank_train(
            start_session_number=args.start,
            end_sesison_number=args.end,
            load_cluster=args.load_cluster,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            negative_k=args.negative_k,
            use_label=args.use_label,
            use_weight=args.use_weight,
            use_tensor_key=args.use_tensor_key,
            max_iters=args.mi,
            warmingup_rate=args.wr,
            init_k=args.init_k,
            cluster_min_size=args.cmnsz,
            sampling_rate=args.sr,
            sampling_size_per_query=args.sspq,
            nbits=args.nbits,
            warming_up_method=args.warming_up_method,
            required_doc_size=args.rdsz,
        )
    elif args.exp == "proposal_ance":
        print(f"Start session: {args.start}")
        print(f"End session: {args.end}")
        print(f"Load cluster: {args.load_cluster}")
        print(f"Use cluster warming up: {args.warming_up_method}")
        print(f"Use label: {args.use_label}")
        print(f"Use weight: {args.use_weight}")
        print(f"Use tensor key: {args.use_tensor_key}")
        print(f"Use nbits: {args.nbits}")
        print(f"Use max_iters: {args.mi}")
        print(f"Use warmingup_rate: {args.wr}")
        print(f"Use warmingup_k: {args.init_k}")
        print(f"Use cluster_min_size: {args.cmnsz}")
        print(f"Use required_doc_size: {args.rdsz}")
        print(f"Use stream sampling rate: {args.sr}")
        print(f"Use sampling size per query: {args.sspq}")
        print(f"Number of Epochs: {args.num_epochs}")
        proposal_ance_train(
            start_session_number=args.start,
            end_sesison_number=args.end,
            load_cluster=args.load_cluster,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            negative_k=args.negative_k,
            use_label=args.use_label,
            use_weight=args.use_weight,
            use_tensor_key=args.use_tensor_key,
            max_iters=args.mi,
            warmingup_rate=args.wr,
            init_k=args.init_k,
            cluster_min_size=args.cmnsz,
            sampling_rate=args.sr,
            sampling_size_per_query=args.sspq,
            nbits=args.nbits,
            warming_up_method=args.warming_up_method,
            required_doc_size=args.rdsz,
        )
    else:
        raise ValueError(f"Unsupported experiments {args.exp}")
