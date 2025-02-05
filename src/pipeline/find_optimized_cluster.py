from functions.utils import draw_elbow
from data import read_jsonl, renew_data
from cluster import kmeans_pp, compute_sse

import torch


def find_best_k(doc_data, start, end, gap, max_iters, devices):
    X = list(doc_data.values())
    print(len(X))
    sse = []
    k_values = range(start, end + 1, gap)
    for _k in k_values:
        centroids, cluster_instances = kmeans_pp(X, _k, max_iters, devices)
        _sse = compute_sse(centroids, cluster_instances, devices)
        sse.append(_sse)
        print(f"K:{_k} | _sse: {_sse}")
    draw_elbow(k_values, sse)


def find_best_k_experiment(start, end, gap, max_iters):
    doc_path = f"../data/sessions/train_session0_docs.jsonl"
    num_gpus = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
    doc_data = read_jsonl(doc_path)[:100]

    doc_count = len(doc_data)
    print(f"Document count:{doc_count}")
    _, doc_data = renew_data(None, doc_data, 24, 768, False, True)
    print(f"doc_data: {len(doc_data)}")

    find_best_k(doc_data, start, end, gap, max_iters, devices)
