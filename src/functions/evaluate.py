from collections import defaultdict
from typing import List

import time
from data import *
from .retriever import get_top_k_documents_by_cosine


def evaluate_dataset(query_path, rankings_path, doc_count, eval_log_path=None):
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

    res_strs = [
        f"# query:  {num_q} |  answers:{total_answer_count} | proportion:{total_answer_count/doc_count * 100:.1f} ",
        f"Avg Success@{top_k_success}: {success / num_q * 100:.1f} ",
        f"Avg Recall@{top_k_recall}: {recall / num_q * 100:.1f} ",
    ]
    print("".join(res_strs))
    if eval_log_path:
        write_lines(eval_log_path, res_strs)


def evaluate_by_cosine(method, sesison_count=1):
    for session_number in range(sesison_count):
        print(f"Evaluate Session {session_number}")
        eval_query_path = (
            f"/mnt/DAIS_NAS/huijeong/test_session{session_number}_queries.jsonl"
        )
        eval_doc_path = (
            f"/mnt/DAIS_NAS/huijeong/test_session{session_number}_docs.jsonl"
        )

        eval_query_data = read_jsonl(eval_query_path)[:10]
        eval_doc_data = read_jsonl(eval_doc_path)[:100]

        eval_query_count = len(eval_query_data)
        eval_doc_count = len(eval_doc_data)
        print(f"Query count:{eval_query_count}, Document count:{eval_doc_count}")

        rankings_path = f"../data/rankings/{method}_session_{session_number}.txt"
        model_path = f"../data/model/{method}_session_{session_number}.pth"

        start_time = time.time()
        new_q_data, new_d_data = renew_data_mean_pooling(
            queries=eval_query_data, documents=eval_doc_data, model_path=model_path
        )
        end_time = time.time()
        print(f"Spend {end_time-start_time} seconds for encoding.")

        start_time = time.time()
        result = get_top_k_documents(new_q_data, new_d_data, k=10)
        end_time = time.time()
        print(f"Spend {end_time-start_time} seconds for retrieval.")

        rankings_path = f"../data/rankings/{method}_session_{session_number}.txt"
        write_file(rankings_path, result)
        evaluate_dataset(eval_query_path, rankings_path, eval_doc_count)
        del new_q_data, new_d_data
