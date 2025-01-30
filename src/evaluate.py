from collections import defaultdict
from typing import List

from .data import *
from .cluster import *
from .functions.utils import process_queries_with_gpus


def evaluate_dataset(query_path, rankings_path, doc_count):
    eval_queries = read_jsonl(query_path)

    rankings = defaultdict(list)
    with open(rankings_path, "r") as f:
        for line in f:
            items = line.strip().split()
            qid: str = items[0]
            pids: List[str] = list(map(str, items[1:]))
            rankings[qid].extend(pids)

    top_k_success, top_k_recall = 5, 10
    success = 0
    num_q = 0
    recall = 0.0
    total_answer_count = 0

    for query in eval_queries:
        num_q += 1
        qid = query["qid"]
        answer_pids = query["answer_pids"]
        total_answer_count += len(answer_pids)

        hit_success = set(rankings[qid][:top_k_success]).intersection(answer_pids)
        if len(hit_success) > 0:
            success += 1

        hit_recall = set(rankings[qid][:top_k_recall]).intersection(answer_pids)
        recall += len(hit_recall) / len(answer_pids)

    print(
        f"# query:  {num_q} |  answers:{total_answer_count} | proportion:{total_answer_count/doc_count * 100:.1f}\n",
        f"Avg Success@{top_k_success}: {success / num_q * 100:.1f}\n",
        f"Avg Recall@{top_k_recall}: {recall / num_q * 100:.1f}\n",
    )


def write_file(rank_file_path, result):
    with open(rank_file_path, "w") as f:
        for key, values in result.items():
            line = f"{key} " + " ".join(map(str, values)) + "\n"
            f.write(line)


def do_retrieval_expermient(query_path, doc_path):
    query_data = read_jsonl(query_path)
    doc_data = read_jsonl(doc_path)

    query_count = len(query_data)
    doc_count = len(doc_data)
    print(f"Query count:{query_count}, Document count:{doc_count}")
    query_data, doc_data = renew_data(query_data, doc_data, 24, 768)

    centroids, cluster_instances = kmeans_pp(list(doc_data.values()), 10, 10, devices)
    result = process_queries_with_gpus(
        query_data, centroids, cluster_instances, devices
    )

    rankings_path = f"./rankings/cluster_retriever.txt"
    write_file(rankings_path, result)
    evaluate_dataset(query_path, rankings_path, doc_count)
