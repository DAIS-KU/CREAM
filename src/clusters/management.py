from typing import List

import torch

import copy
from data import Stream
from functions import calculate_S_qd_regl_dict, get_passage_embeddings

from .cluster import Cluster
from .clustering import kmeans_pp
from .encode import renew_data
from .prototype import RandomProjectionLSH

from concurrent.futures import ThreadPoolExecutor

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


def find_k_closest_clusters(
    model, texts: List[str], clusters: List[Cluster], k, device
) -> List[int]:
    token_embs = get_passage_embeddings(model, texts, device)
    scores = []
    for cluster in clusters:
        score = calculate_S_qd_regl_dict(token_embs, cluster.prototype, device)
        scores.append(score.unsqueeze(1))
    scores_tensor = torch.cat(scores, dim=1)
    topk_values, topk_indices = torch.topk(
        scores_tensor, k, dim=1
    )  # 각 샘플별 k개 선택
    return topk_indices.tolist()


def assign_instance_or_add_cluster(
    model,
    lsh: RandomProjectionLSH,
    clusters: List[Cluster],
    stream_docs,
    docs: dict,
    ts,
    batch_size=64,
):
    print("assign_instance_or_add_cluster started.")
    num_devices = len(devices)
    batches = [
        stream_docs[i : i + batch_size] for i in range(0, len(stream_docs), batch_size)
    ]

    def process_batch(batch, device):
        print(f"ㄴ Batch {len(batch)} starts on {device}.")
        temp_model = copy.deepcopy(model).to(device)

        texts = [doc["text"] for doc in batch]
        doc_ids = [doc["doc_id"] for doc in batch]
        batch_embs = get_passage_embeddings(temp_model, texts, device).squeeze()
        batch_closest_ids = find_k_closest_clusters(
            temp_model, texts, clusters, 1, device
        )

        for i, doc in enumerate(batch):
            closest_cluster_id = batch_closest_ids[i][0]
            closest_cluster = clusters[closest_cluster_id]
            doc_embs = batch_embs[i]

            s1, s2, n = closest_cluster.get_statistics()
            doc_hash = lsh.encode(doc_embs)
            closest_distance = closest_cluster.get_distance(doc_embs)
            closest_boundary = closest_cluster.get_boundary()

            if n == 1 or closest_distance <= closest_boundary:
                closest_cluster.assign(doc_ids[i], doc_embs, doc_hash, ts)
            else:
                clusters.append(Cluster(temp_model, doc_hash, [doc], docs, ts))

    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        futures = []
        for i, batch in enumerate(batches):
            device = devices[i % num_devices]
            futures.append(executor.submit(process_batch, batch, device))
        for future in futures:
            future.result()

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
    chunk_size = (len(clusters) + num_devices - 1) // num_devices
    cluster_chunks = [
        clusters[i : i + chunk_size] for i in range(0, len(clusters), chunk_size)
    ]

    def process_cluster_chunk(cluster_chunk, device):
        local_model = copy.deepcopy(model).to(device)
        local_result = []
        for cluster in cluster_chunk:
            is_alive = (
                True
                if cluster.timestamp >= ts
                else cluster.evict(local_model, lsh, docs)
            )
            if is_alive:
                local_result.append(cluster)
        return local_result

    remaining_clusters = []
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        futures = {
            executor.submit(
                process_cluster_chunk, cluster_chunk, devices[i % num_devices]
            ): cluster_chunk
            for i, cluster_chunk in enumerate(cluster_chunks)
        }
        for future in futures:
            remaining_clusters.extend(future.result())

    print(f"evict_cluster_instances finished. (left #{len(remaining_clusters)})")
    return remaining_clusters


def retrieve_top_k_docs_from_cluster(
    model, queries, clusters, docs: dict, k=10, batch_size=64
):
    print("retrieve_top_k_docs_from_cluster started.")
    result = {}
    batches = [queries[i : i + batch_size] for i in range(0, len(queries), batch_size)]

    def process_batch(batch, device):
        print(f"ㄴ Batch {len(batch)} starts on {device}.")
        batch_result = {}
        temp_model = copy.deepcopy(model).to(device)

        qids = [query["qid"] for query in batch]
        texts = [query["query"] for query in batch]
        batch_closest_ids = find_k_closest_clusters(
            temp_model, texts, clusters, 1, device
        )

        for i, query in enumerate(batch):
            closest_cluster_id = batch_closest_ids[i][0]
            closest_cluster = clusters[closest_cluster_id]
            top_k_doc_ids = closest_cluster.get_topk_docids(temp_model, query, docs, k)
            batch_result[qids[i]] = top_k_doc_ids
        return batch_result

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(process_batch, batch, devices[i % num_threads]): batch
            for i, batch in enumerate(batches)
        }
        for future in futures:
            result.update(future.result())

    print("retrieve_top_k_docs_from_cluster finished.")
    return result
