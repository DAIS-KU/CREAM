from functions import (
    calculate_S_qd_regl,
    read_jsonl_as_dict,
    read_jsonl,
    save_jsonl,
    renew_data_mean_pooling,
)
from clusters import RandomProjectionLSH, renew_data
from functions import get_top_k_documents, evaluate_dataset
from data import write_file

import random
import torch
import torch.nn.functional as F
from transformers import BertModel
import numpy as np

import time

num_gpus = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]


def sampling_negs(pos_ids, doc_ids):
    neg_ids = random.sample(
        [doc_id for doc_id in doc_ids if doc_id not in pos_ids], len(pos_ids)
    )
    return neg_ids


def generate_pooling_data(qnt=200):
    source_path = "/home/work/.default/huijeong/data/lotte"
    dest_path = "/home/work/.default/huijeong/data/pooled"
    domains = ["technology", "writing", "science", "recreation", "lifestyle"]
    # domains = ["domain0", "domain1", "domain2", "domain3", "domain4"]
    pooled_query_path = f"{dest_path}/pooled_queries.jsonl"
    pooled_docs_path = f"{dest_path}/pooled_docs.jsonl"
    pooled_queries = []
    pooled_docs = []
    for domain in domains:
        print(f"Processing domain: {domain}")
        query_path = f"{source_path}/{domain}/{domain}_queries.jsonl"
        docs_path = f"{source_path}/{domain}/{domain}_docs.jsonl"
        queries = read_jsonl(query_path, True)
        queries = random.sample(queries, qnt)
        docs = read_jsonl_as_dict(docs_path, "doc_id")
        answer_docs = []
        for query in queries:
            answer_doc_ids = query["answer_pids"]
            answer_docs.extend(docs[doc_id] for doc_id in answer_doc_ids)
        pooled_queries.extend(queries)
        pooled_docs.extend(answer_docs)

    pooled_doc_ids = [doc["doc_id"] for doc in pooled_docs]
    for query in pooled_queries:
        neg_ids = sampling_negs(query["answer_pids"], pooled_doc_ids)
        query["neg_pids"] = neg_ids
    save_jsonl(pooled_queries, pooled_query_path)
    save_jsonl(pooled_docs, pooled_docs_path)


def model_builder(model_path):
    model = BertModel.from_pretrained("/home/work/.default/huijeong/bert_local").to(
        devices[-1]
    )
    if model_path:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def get_cosine_scores(q, d):
    q_norm = F.normalize(q, p=2, dim=0)
    d_norm = F.normalize(d, p=2, dim=0)
    return torch.sum(q_norm * d_norm).item()


import numpy as np


def get_mean_pooling_success_recall(k=5):
    path = "/home/work/.default/huijeong/data/lotte_session"
    queries = read_jsonl_as_dict(f"{path}/test_session0_queries.jsonl", "qid")
    docs = read_jsonl(f"{path}/train_session0_docs.jsonl", False)
    new_q_data, new_d_data = renew_data_mean_pooling(
        model_builder, None, list(queries.values()), docs
    )
    all_doc_ids = list(new_d_data.keys())
    all_doc_embs = [new_d_data[did]["EMB"] for did in all_doc_ids]

    success_list = []
    recall_list = []

    for q in new_q_data.values():
        qid = q["doc_id"]
        q_emb = q["EMB"]
        scores = [get_cosine_scores(q_emb, d_emb) for d_emb in all_doc_embs]

        order = np.argsort(-np.asarray(scores))
        topk_idx = order[: min(k, len(order))]
        topk_ids = {all_doc_ids[i] for i in topk_idx}

        pos_ids = set(queries[qid].get("answer_pids", []))
        n_pos = len(pos_ids)
        if n_pos == 0:
            continue  # 정답이 없는 질의는 평균에서 제외

        # Success@k (hit)
        hit = 1 if len(pos_ids & topk_ids) > 0 else 0
        success_list.append(hit)
        # Recall@k
        recall_k = len(pos_ids & topk_ids) / n_pos
        recall_list.append(recall_k)

    success_mean = float(np.mean(success_list)) if success_list else float("nan")
    recall_mean = float(np.mean(recall_list)) if recall_list else float("nan")
    return {"success@k_mean": success_mean, "recall@k_mean": recall_mean}


def evaluate(nbits, session_count=1):
    for session_number in range(session_count):
        _evaluate(nbits, session_number)


def _evaluate(nbits, session_number):
    method = f"lsh_msmarco_{nbits}"
    print(f"Evaluate Session {session_number}")
    eval_query_path = f"/home/work/.default/huijeong/data/lotte_session/test_session{session_number}_queries.jsonl"
    eval_doc_path = f"/home/work/.default/huijeong/data/lotte_session/train_session{session_number}_docs.jsonl"
    eval_query_data = read_jsonl(eval_query_path, True)
    eval_doc_data = read_jsonl(eval_doc_path, False)

    eval_query_count = len(eval_query_data)
    eval_doc_count = len(eval_doc_data)
    print(f"Query count:{eval_query_count}, Document count:{eval_doc_count}")

    rankings_path = f"/home/work/.default/huijeong/data/rankings/{method}_session_{session_number}.txt"

    start_time = time.time()
    new_q_data, new_d_data = renew_data(
        queries=eval_query_data,
        documents=eval_doc_data,
        model_path=None,
        nbits=nbits,
        renew_q=True,
        renew_d=True,
        use_tensor_key=True,
    )
    end_time = time.time()
    print(f"Spend {end_time-start_time} seconds for encoding.")

    start_time = time.time()
    result = get_top_k_documents(new_q_data, new_d_data)
    end_time = time.time()
    print(f"Spend {end_time-start_time} seconds for retrieval.")

    rankings_path = f"/home/work/.default/huijeong/data/rankings/{method}_session_{session_number}.txt"
    write_file(rankings_path, result)
    eval_log_path = (
        f"/home/work/.default/huijeong/data/evals/{method}_{session_number}.txt"
    )
    evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
    del new_q_data, new_d_data


def compare(generate_data=False):
    # if generate_data:
    #     generate_pooling_data()
    # summary = get_mean_pooling_success_recall(k=10)
    # print(f"summary(mean): {summary}\n")

    # evaluate(nbits=0)

    evaluate(nbits=3)

    evaluate(nbits=6)

    evaluate(nbits=9)

    evaluate(nbits=12)
