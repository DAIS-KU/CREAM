import time
from collections import defaultdict
from typing import List

from data import *

from .retriever import get_top_k_documents_by_cosine


# def evaluate_dataset(query_path, rankings_path, doc_count, eval_log_path=None):
#     eval_queries = read_jsonl(query_path, True)

#     rankings = defaultdict(list)
#     with open(rankings_path, "r") as f:
#         for line in f:
#             items = line.strip().split()
#             qid: str = items[0]
#             pids: List[str] = list(map(str, items[1:]))
#             rankings[qid].extend(pids)

#     top_k_success, top_k_recall = 5, 10
#     success = 0
#     num_q = 0
#     recall = 0.0
#     total_answer_count = 0

#     for query in eval_queries:
#         num_q += 1
#         qid = query["qid"]
#         answer_pids = query["answer_pids"]
#         total_answer_count += len(answer_pids)

#         hit_success = set(rankings[qid][:top_k_success]).intersection(answer_pids)
#         if len(hit_success) > 0:
#             success += 1

#         hit_recall = set(rankings[qid][:top_k_recall]).intersection(answer_pids)
#         recall += len(hit_recall) / len(answer_pids)

#     res_strs = [
#         f"# query:  {num_q} |  answers:{total_answer_count} | proportion:{total_answer_count/doc_count * 100:.1f} ",
#         f"Avg Success@{top_k_success}: {success / num_q * 100:.1f} ",
#         f"Avg Recall@{top_k_recall}: {recall / num_q * 100:.1f} ",
#     ]
#     print("".join(res_strs))
#     if eval_log_path:
#         write_lines(eval_log_path, res_strs)


def evaluate_dataset(query_path, rankings_path, doc_count, eval_log_path=None):
    eval_queries = read_jsonl(query_path, True)
    rankings = defaultdict(list)
    with open(rankings_path, "r") as f:
        for line in f:
            items = line.strip().split()
            qid: str = items[0]
            pids: List[str] = list(map(str, items[1:]))
            rankings[qid].extend(pids)

    top_k_success, top_k_recall = 5, 10
    domain_prefixes = [
        "writing",
        "technology",
        "lifestyle",
        "science",
        "recreation",
        "domain0",
        "domain1",
        "domain2",
        "domain3",
        "domain4",
    ]
    domain_stats = {
        prefix: {"success": 0, "recall": 0.0, "total": 0, "answer_count": 0}
        for prefix in domain_prefixes
    }
    overall_success = 0
    overall_recall = 0.0
    overall_total = 0
    total_answer_count = 0

    for query in eval_queries:
        qid = query["qid"]
        answer_pids = query["answer_pids"]
        total_answer_count += len(answer_pids)
        overall_total += 1

        hit_success = set(rankings[qid][:top_k_success]).intersection(answer_pids)
        hit_recall = set(rankings[qid][:top_k_recall]).intersection(answer_pids)

        if len(hit_success) > 0:
            overall_success += 1
        else:
            print(f"QID {qid} is totally wrong~!")
        overall_recall += len(hit_recall) / len(answer_pids)

        for prefix in domain_prefixes:
            if qid.startswith(prefix):
                domain_stats[prefix]["total"] += 1
                domain_stats[prefix]["answer_count"] += len(answer_pids)
                if len(hit_success) > 0:
                    domain_stats[prefix]["success"] += 1
                domain_stats[prefix]["recall"] += len(hit_recall) / len(answer_pids)
                break

    res_strs = []
    res_strs.append(
        f"# query: {overall_total} | answers: {total_answer_count} | proportion: {total_answer_count / doc_count * 100:.1f}%"
    )
    res_strs.append(
        f"Avg Success@{top_k_success}: {overall_success / overall_total * 100:.1f}%"
    )
    res_strs.append(
        f"Avg Recall@{top_k_recall}: {overall_recall / overall_total * 100:.1f}%"
    )
    res_strs.append("")

    for prefix in domain_prefixes:
        stats = domain_stats[prefix]
        if stats["total"] > 0:
            res_strs.append(f"[{prefix}]")
            res_strs.append(
                f"  # query: {stats['total']} | answers: {stats['answer_count']} | proportion: {stats['answer_count'] / doc_count * 100:.1f}%"
            )
            res_strs.append(
                f"  Avg Success@{top_k_success}: {stats['success'] / stats['total'] * 100:.1f}%"
            )
            res_strs.append(
                f"  Avg Recall@{top_k_recall}: {stats['recall'] / stats['total'] * 100:.1f}%"
            )
            res_strs.append("")

    print("\n".join(res_strs))
    if eval_log_path:
        write_lines(eval_log_path, res_strs)
