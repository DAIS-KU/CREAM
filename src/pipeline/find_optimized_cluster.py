from functions.utils import draw_elbow
from data import read_jsonl, renew_data
from cluster import kmeans_pp, compute_sse

import torch

import time


import time


def find_best_k(doc_data, start, end, gap, max_iters, devices):
    X = list(doc_data.values())
    print(len(X))
    sse, execution_times = [], []
    k_values = range(start, end + 1, gap)
    filename = "./find_k.txt"
    with open(filename, "a") as file:
        for _k in k_values:
            start_time = time.time()
            centroids, cluster_instances = kmeans_pp(X, _k, max_iters, devices)
            _sse = compute_sse(centroids, cluster_instances, devices)
            sse.append(_sse)
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)

        for k, _sse, et in zip(k_values, sse, execution_times):
            result = f"k {k}: sse {_sse} ({et} sec.)\n"
            print(result)
            file.write(result)


def find_best_k_experiment(start, end, gap, max_iters):
    doc_path = f"/mnt/DAIS_NAS/huijeong/train_session0_docs.jsonl"
    num_gpus = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
    doc_data = read_jsonl(doc_path)

    doc_count = len(doc_data)
    print(f"Document count:{doc_count}")
    _, doc_data = renew_data(
        queries=None,
        documents=doc_data,
        nbits=16,
        embedding_dim=768,
        model_path=None,
        renew_q=False,
        renew_d=True,
    )
    print(f"doc_data: {len(doc_data)}")

    find_best_k(doc_data, start, end, gap, max_iters, devices)
