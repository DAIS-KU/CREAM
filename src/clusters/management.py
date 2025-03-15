import copy
from concurrent.futures import ThreadPoolExecutor
from typing import List

import torch

from data import Stream
from functions import (
    calculate_S_qd_regl_dict,
    get_passage_embeddings,
    calculate_S_qd_regl_batch,
    calculate_S_qd_regl_batch_batch,
)
import numpy as np

from .cluster import Cluster
from .clustering import kmeans_pp
from .tensor_clustering import kmeans_pp_use_tensor_key
from .encode import renew_data
from .prototype import RandomProjectionLSH

num_devices = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_devices)]


def initialize(
    model, stream: Stream, k, nbits, max_iters, use_tensor_key=True
) -> List[Cluster]:
    _, initial_docs = renew_data(
        queries=None,
        documents=stream.initial_docs,
        renew_q=False,
        renew_d=True,
        nbits=nbits,
        use_tensor_key=use_tensor_key,
    )
    if use_tensor_key:
        centroids, cluster_instances = kmeans_pp_use_tensor_key(
            list(initial_docs.values()), k, max_iters, nbits
        )
    else:
        centroids, cluster_instances = kmeans_pp(
            list(initial_docs.values()), k, max_iters, nbits
        )

    clusters = []
    for cid, centroid in enumerate(centroids):
        if len(cluster_instances[cid]):
            print(f"Create {len(clusters)}th Cluster.")
            clusters.append(
                Cluster(
                    model, centroid, cluster_instances[cid], stream.docs, use_tensor_key
                )
            )
    return clusters


# tensor_clustering.get_closest_clusters_use_tensor_key
def find_k_closest_clusters(
    model,
    texts: List[str],
    clusters: List[Cluster],  # 얘가 왔다갔다해서 느린가..?
    k,
    device,
    use_tensor_key,
    batch_size=8,
) -> List[int]:
    token_embs = get_passage_embeddings(model, texts, device)
    prototypes = [cluster.prototype for cluster in clusters]
    scores = []
    if use_tensor_key:
        for i in range(0, len(prototypes), batch_size):
            batch_prototypes = prototypes[i : i + batch_size]  # batch_size만큼 자르기
            batch_prototypes = torch.stack(prototypes[i : i + batch_size]).to(device)
            # print(
            #     f"token_embs:{token_embs.shape}, batch_prototypes:{batch_prototypes.shape}"
            # )
            batch_scores = calculate_S_qd_regl_batch_batch(
                token_embs, batch_prototypes, device
            )
            scores.append(batch_scores)
    else:
        for prototype in prototypes:
            score = calculate_S_qd_regl_dict(token_embs, prototype, device)
            scores.append(score.unsqueeze(1))
    scores_tensor = torch.cat(scores, dim=1)
    topk_values, topk_indices = torch.topk(
        scores_tensor, k, dim=1
    )  # 각 샘플별 k개 선택
    # print(
    #     f"find_k_closest_clusters scores: {scores_tensor.shape}, topk_indices:{topk_indices.shape}"
    # )
    return topk_indices.tolist()


def find_k_closest_clusters_for_sampling(
    model,
    texts: List[str],
    clusters: List[Cluster],
    k,
    use_tensor_key,
    batch_size=8,
) -> List[int]:
    token_embs = get_passage_embeddings(model, texts, devices[1])
    prototypes = [cluster.prototype for cluster in clusters]
    scores = []
    if use_tensor_key:
        # not stride, 순서 보장 필요
        prototype_batches = list(map(list, np.array_split(prototypes, num_devices)))

        def process_on_device(device, batch_prototypes):
            device_scores = []
            temp_token_embs = token_embs.clone().to(device)
            for i in range(0, len(batch_prototypes), batch_size):
                batch = batch_prototypes[i : i + batch_size]
                batch_tensor = torch.stack(batch).to(device)
                batch_scores = calculate_S_qd_regl_batch_batch(
                    temp_token_embs, batch_tensor, device
                ).cpu()
                device_scores.append(batch_scores)
            return (
                torch.cat(device_scores, dim=1)
                if device_scores
                else torch.empty(0, len(token_embs))
            )

        with ThreadPoolExecutor(num_devices) as executor:
            # map 결과 순서 보장
            scores = list(executor.map(process_on_device, devices, prototype_batches))
    else:
        for prototype in prototypes:
            score = calculate_S_qd_regl_dict(token_embs, prototype, devices[1])
            scores.append(score.unsqueeze(1))
    scores_tensor = torch.cat(scores, dim=1)
    topk_values, topk_indices = torch.topk(
        scores_tensor, k, dim=1
    )  # 각 샘플별 k개 선택
    # print(
    #     f"find_k_closest_clusters_for_sampling scores: {scores_tensor.shape}, topk_indices:{topk_indices.shape}"
    # )
    return topk_indices.tolist()


def assign_instance_or_add_cluster(
    model,
    lsh: RandomProjectionLSH,
    clusters: List[Cluster],
    stream_docs,
    docs: dict,
    ts,
    use_tensor_key,
    cluster_min_size,
):
    print("assign_instance_or_add_cluster started.")
    # stride. 순서 보장 필요X
    batch_cnt = min(num_devices, len(stream_docs))
    batches = [stream_docs[i::batch_cnt] for i in range(batch_cnt)]

    def process_batch(batch, device):
        print(f"ㄴ Batch {len(batch)} starts on {device}.")
        temp_model = copy.deepcopy(model).to(device)
        texts = [doc["text"] for doc in batch]
        doc_ids = [doc["doc_id"] for doc in batch]
        batch_embs = get_passage_embeddings(temp_model, texts, device).squeeze()
        batch_closest_ids = find_k_closest_clusters(
            temp_model, texts, clusters, 1, device, lsh.use_tensor_key
        )
        # TODO 모든 인스턴스별 로직(병목), cpu작업(비동기 소용x) 우야노............더 못 줄이겠는데
        # Str
        for i, doc in enumerate(batch):
            closest_cluster_id = batch_closest_ids[i][0]
            closest_cluster = clusters[closest_cluster_id]
            doc_embs = batch_embs[i]

            s1, s2, n = closest_cluster.get_statistics()
            doc_hash = lsh.encode(doc_embs)
            closest_distance = closest_cluster.get_distance(doc_embs)
            closest_boundary = closest_cluster.get_boundary()
            if n <= cluster_min_size or closest_distance <= closest_boundary:
                closest_cluster.assign(doc_ids[i], doc_embs, doc_hash, ts)
            else:
                # 왜 이렇게 많지..? 그냥 저냥 납득할만한 거리인 것 같기도 하고..
                # 작은 클러스터 반경이 넘 작아서 그런가?  초기화 때 많이 만들어 놓는게 안정적인가? <- 그런듯 <- 꼭 그렇지만도 안음,,흑,,,,,,,,,,
                # n <= min_size 이런식으로 좀 묶어줘야하나 뒤로 갈수록 많이 생기는 거 같은데
                print(f"closest_cluster: {s1}, {s2}, {n}")
                print(
                    f"closest_distance: {closest_distance}, closest_boundary:{closest_boundary}"
                )
                clusters.append(
                    Cluster(temp_model, doc_hash, [doc], docs, use_tensor_key, ts)
                )

    with ThreadPoolExecutor(max_workers=batch_cnt) as executor:
        futures = []
        for i, batch in enumerate(batches):
            device = devices[i % batch_cnt]
            futures.append(executor.submit(process_batch, batch, device))
        for future in futures:
            future.result()

    print(f"assign_instance_or_add_cluster finished.({len(clusters)})")
    return clusters


def get_samples_and_weights(
    model, query, docs: dict, clusters, positive_k, negative_k, ts, use_tensor_key
):
    cluster_ids = find_k_closest_clusters_for_sampling(
        model=model,
        texts=[query["query"]],
        clusters=clusters,
        k=positive_k + negative_k,
        use_tensor_key=use_tensor_key,
    )
    positive_id = cluster_ids[0][0]
    negative_ids = cluster_ids[0][1:]
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
    # stride. 순서 보장 필요X
    cluster_chunks = [clusters[i::num_devices] for i in range(num_devices)]

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
    model, queries, clusters, docs: dict, use_tensor_key, k=10, batch_size=64
):
    print("retrieve_top_k_docs_from_cluster started.")
    result = {}
    # stride. 순서 보장 필요X
    batches = [queries[i::num_devices] for i in range(num_devices)]

    def process_batch(batch, device):
        print(f"ㄴ Batch {len(batch)} starts on {device}.")
        batch_result = {}
        temp_model = copy.deepcopy(model).to(device)

        qids = [query["qid"] for query in batch]
        texts = [query["query"] for query in batch]
        batch_closest_ids = find_k_closest_clusters(
            temp_model, texts, clusters, 1, device, use_tensor_key
        )

        for i, query in enumerate(batch):
            closest_cluster_id = batch_closest_ids[i][0]
            closest_cluster = clusters[closest_cluster_id]
            top_k_doc_ids = closest_cluster.get_topk_docids(temp_model, query, docs, k)
            batch_result[qids[i]] = top_k_doc_ids
        return batch_result

    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        futures = {
            executor.submit(process_batch, batch, devices[i % num_devices]): batch
            for i, batch in enumerate(batches)
        }
        for future in futures:
            result.update(future.result())

    print("retrieve_top_k_docs_from_cluster finished.")
    return result
