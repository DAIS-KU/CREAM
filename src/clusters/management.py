from typing import List

import torch

from data import Stream
from functions import calculate_S_qd_regl_dict, get_passage_embeddings

from .cluster import Cluster
from .clustering import kmeans_pp
from .encode import renew_data
from .prototype import RandomProjectionLSH

num_devices = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_devices)]


def initialize(model, stream: Stream, k, nbits, max_iters=5) -> List[Cluster]:
    _, initial_docs = renew_data(
        queries=None,
        documents=stream.initial_docs,
        nbits=nbits,
        embedding_dim=768,
        renew_q=False,
        renew_d=True,
    )
    centroids, cluster_instances = kmeans_pp(list(initial_docs.values()), k, max_iters)
    clusters = []
    for cid, centroid in enumerate(centroids):
        print(f"Create {cid}th Cluster.")
        clusters.append(Cluster(model, centroid, cluster_instances[cid], stream.docs))
    return clusters


def find_k_closest_cluster(model, text, clusters: List[Cluster], k) -> List[int]:
    token_embs = get_passage_embeddings(model, text, devices[-1])
    distances = []
    for cluster in clusters:
        distances.append(
            calculate_S_qd_regl_dict(token_embs, cluster.prototype, devices[-1])
        )
    sorted_distances, sorted_indices = torch.sort(
        torch.stack(distances).squeeze(), descending=True
    )
    return sorted_indices[:k].tolist()


def assign_instance_or_add_cluster(
    model,
    lsh: RandomProjectionLSH,
    clusters: List[Cluster],
    stream_docs,
    docs: dict,
    ts,
):
    print("assign_instance_or_add_cluster started.")
    for i, doc in enumerate(stream_docs):
        print(f"- {i}th stream doc.")
        closest_cluster_id = find_k_closest_cluster(model, doc["text"], clusters, 1)[0]
        closest_clsuter = clusters[closest_cluster_id]
        doc_embs = get_passage_embeddings(model, doc["text"], devices[-1]).squeeze()

        # 데이터 포인트 1개일 때 통계정보 없으므로 무조건 할당
        # 2개 이상부터 RMS이내인지 학인, 아니면 새로운 centroid으로 할당
        s1, s2, n = closest_clsuter.get_statistics()
        doc_hash = lsh.encode(doc_embs)
        closest_distance = closest_clsuter.get_distance(doc_embs)
        closest_boundary = closest_clsuter.get_boundary()

        if n == 1 or closest_distance <= closest_boundary:
            closest_clsuter.assign(doc["doc_id"], doc_embs, doc_hash, ts)
        else:
            clusters.append(Cluster(model, doc_hash, [doc], docs, ts))
    print("assign_instance_or_add_cluster finished.")
    return clusters


def get_samples_and_weights(
    model, query, docs: dict, clusters, positive_k, negative_k, ts
):
    cluster_ids = find_k_closest_cluster(
        model, query["query"], clusters, positive_k + negative_k
    )
    positive_id = cluster_ids[0]
    negative_ids = cluster_ids[1:]
    print(f"positive_id:{positive_id} | negative_ids:{negative_ids}")

    positive_cluster = clusters[positive_id]
    positive_samples = positive_cluster.get_topk_docids(model, query, docs, 1)
    positive_weights = [positive_cluster.get_weight(ts)]

    negative_samples, negative_weights = [], []
    for neg_id in negative_ids:
        negative_cluster = clusters[neg_id]
        neg_docs = negative_cluster.get_topk_docids(model, query, docs, 1)
        negative_samples.extend(neg_docs)
        negative_weights.append(negative_cluster.get_weight(ts))

    print(
        f" query: {query['qid']} | positive: {positive_samples} | negative:{negative_samples}"
    )
    return positive_samples, positive_weights, negative_samples, negative_weights


def evict_clusters(
    model, lsh, docs: dict, clusters: List[Cluster], ts
) -> List[Cluster]:
    print("evict_cluster_instances started.")
    masks = []
    for i, cluster in enumerate(clusters):
        is_alive = True if cluster.timestamp >= ts else cluster.evict(model, lsh, docs)
        masks.append(is_alive)
    clusters = [clusters[i] for i in range(len(clusters)) if masks[i]]
    print(f"evict_cluster_instances finished. (left #{len(clusters)})")
    return clusters


def retrieve_top_k_docs_from_cluster(model, queries, clusters, docs: dict, k=10):
    print("retrieve_top_k_docs_from_cluster started.")
    result = {}
    for qid, query in enumerate(queries):
        print(f"- retrieve {qid}th query")
        closest_cluster_id = find_k_closest_cluster(model, query["query"], clusters, 1)[
            0
        ]
        closest_cluster = clusters[closest_cluster_id]
        top_k_doc_ids = closest_cluster.get_topk_docids(model, query, docs, k)
        {query["qid"]: top_k_doc_ids}
    print("retrieve_top_k_docs_from_cluster finished.")
    return result
