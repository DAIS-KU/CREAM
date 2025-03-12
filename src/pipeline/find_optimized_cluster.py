import random
import time

import numpy as np
import torch

from clusters import compute_sse, kmeans_pp, renew_data
from data import read_jsonl
from functions.utils import draw_elbow


def find_best_k(doc_data, max_iters, devices):
    X = list(doc_data.values())
    print(len(X))
    sse, execution_times = [], []
    max_k = np.sqrt(len(X) / 2)
    k_values = [round(k) for k in np.linspace(max_k // 2, max_k, 10)]
    filename = "./find_k.txt"
    with open(filename, "a") as file:
        for _k in k_values:
            start_time = time.time()
            centroids, cluster_instances = kmeans_pp(X, _k, max_iters)
            _sse = compute_sse(centroids, cluster_instances)
            sse.append(_sse)
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)

        for k, _sse, et in zip(k_values, sse, execution_times):
            result = f"k {k}: sse {_sse} ({et} sec.)\n"
            print(result)
            file.write(result)


def find_best_k_experiment(max_iters=5, warmingup_rate=0.25):
    num_gpus = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]

    doc_path = f"/mnt/DAIS_NAS/huijeong/train_session0_docs.jsonl"
    doc_data = read_jsonl(doc_path, False)
    wr_cnt = int(len(doc_data) * warmingup_rate)
    random.shuffle(doc_data)
    doc_data = doc_data[:wr_cnt]

    doc_count = len(doc_data)
    print(f"Document count:{doc_count}, devices:{devices}")
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

    find_best_k(doc_data, max_iters, devices)
