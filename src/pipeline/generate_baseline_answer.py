import json
import random
import torch
from collections import Counter, defaultdict
from functions import renew_data_mean_pooling, get_top_k_documents_by_cosine
from data import read_jsonl
from transformers import BertModel
import os

num_gpus = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]


def append_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def model_builder(model_path):
    model = BertModel.from_pretrained("bert-base-uncased").to(devices[-1])
    return model


def create_cos_ans_file(dataset="datasetM"):
    train_query_path = (
        f"/home/work/retrieval/data/{dataset}/train_session0_queries.jsonl"
    )
    train_docs_path = f"/home/work/retrieval/data/{dataset}/train_session0_docs.jsonl"
    queries_data = read_jsonl(train_query_path, True)
    documents_data = read_jsonl(train_docs_path, False)
    new_q_data, new_d_data = renew_data_mean_pooling(
        model_builder=model_builder,
        model_path=None,
        queries_data=queries_data,
        documents_data=documents_data,
    )
    qid_kpids = get_top_k_documents_by_cosine(
        new_q_data=new_q_data, new_d_data=new_d_data, k=1, batch_size=4096
    )

    target_path = (
        f"/home/work/retrieval/data/{dataset}/train_session0_queries_cos.jsonl"
    )
    for q in queries_data:
        q["cos_ans_pids"] = qid_kpids[q["qid"]]
    append_jsonl(queries_data, target_path)
