import copy
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any

import numpy as np
import torch
import torch.nn.functional as F

from .stream import Stream
from functions import (
    calculate_S_qd_regl_batch,
    calculate_S_qd_regl_batch_batch,
    calculate_S_qd_regl_dict,
)
import threading

from .cluster import Cluster
from .clustering import kmeans_pp
from .encode import renew_data
from .prototype import RandomProjectionLSH
from .tensor_clustering import kmeans_pp_use_tensor_key
from data import BM25Okapi, preprocess

num_devices = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_devices)]
MAX_SCORE = 256


def initialize(
    stream_docs, docs, k, nbits, lsh, max_iters, use_tensor_key=True
) -> List[Cluster]:
    # _, enoded_stream_docs =  renew_data(
    #     queries=None,
    #     documents=stream_docs,
    #     renew_q=False,
    #     renew_d=True,
    #     nbits=nbits,
    #     use_tensor_key=use_tensor_key,
    #     model_path="../data/base_model_lotte.pth",
    # )
    # enoded_stream_docs = list(enoded_stream_docs.values())
    enoded_stream_docs = stream_docs
    # print(f"enoded_stream_docs: {enoded_stream_docs[0]}")
    centroids, cluster_instances = kmeans_pp_use_tensor_key(
        enoded_stream_docs, k, max_iters, nbits
    )
    clusters = []
    for cid, centroid in enumerate(centroids):
        if len(cluster_instances[cid]):
            print(f"Create {len(clusters)}th Cluster.")
            clusters.append(
                Cluster(centroid, cluster_instances[cid], docs, use_tensor_key)
            )
    return clusters


# tensor_clustering.get_closest_clusters_use_tensor_key
def find_k_closest_clusters(
    token_embs,
    clusters: List[Cluster],
    k,
    device,
    use_tensor_key,
    batch_size=4,
    return_distance=False,
) -> List[int]:
    if not isinstance(token_embs, torch.Tensor):
        token_embs = torch.stack(token_embs, dim=0)
    prototypes = [cluster.prototype for cluster in clusters]
    scores = []
    if use_tensor_key:
        for i in range(0, len(prototypes), batch_size):
            batch_prototypes = torch.stack(prototypes[i : i + batch_size]).to(device)
            batch_scores = calculate_S_qd_regl_batch_batch(
                token_embs, batch_prototypes, device
            ).cpu()  # (t_bsz, p_bsz)
            scores.append(batch_scores)
    else:
        for prototype in prototypes:
            score = calculate_S_qd_regl_dict(token_embs, prototype, device)
            scores.append(score.unsqueeze(1))
    if len(scores) == 0:
        print(f"prototypes: {len(prototypes)}")
    scores_tensor = torch.cat(scores, dim=1)  # (t_bsz, len(prototypes))
    topk_values, topk_indices = torch.topk(
        scores_tensor, k, dim=1
    )  # 각 샘플별 k개 선택
    # print(
    #     f"scores: {scores_tensor.shape}, topk_indices:{topk_indices.shape}"
    # )
    if return_distance:
        topk_distances = MAX_SCORE - topk_values
        return topk_distances.tolist(), topk_indices.tolist()
    else:
        return topk_indices.tolist()


def find_k_closest_clusters_for_sampling(
    token_embs: List[Any],
    clusters: List[Cluster],
    k,
    use_tensor_key=True,
    batch_size=8,
) -> List[int]:
    token_embs = torch.stack(token_embs, dim=0)
    prototype_tensor = torch.stack([cluster.prototype for cluster in clusters])
    scores = []
    # prototype_batches = list(map(list, np.array_split(prototypes, num_devices)))
    prototype_batches = list(
        torch.chunk(prototype_tensor, num_devices)
    )  # 각 chunk는 Tensor
    prototype_batches = [
        list(batch) for batch in prototype_batches
    ]  # 각 batch를 리스트로 변환

    def process_on_device(device, batch_prototypes):
        device_scores = []
        temp_token_embs = token_embs.clone().to(device)
        for i in range(0, len(batch_prototypes), batch_size):
            batch_tensor = torch.stack(batch_prototypes[i : i + batch_size]).to(device)
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
        scores = list(
            executor.map(process_on_device, devices, prototype_batches)
        )  # map 결과 순서 보장
    scores_tensor = torch.cat(scores, dim=1)
    topk_values, topk_indices = torch.topk(
        scores_tensor, k, dim=1
    )  # 각 샘플별 k개 선택
    bottomk_values, bottomk_indices = torch.topk(
        scores_tensor, k, dim=1, largest=False
    )  # 각 샘플별 k개 선택
    # print(
    #     f"scores: {scores_tensor.shape}, topk_indices:{topk_indices.shape}"
    # )
    return topk_indices.tolist(), bottomk_indices.tolist()


def assign_instance_or_add_cluster(
    lsh,
    clusters,
    stream_docs,
    docs,
    ts,
    use_tensor_key,
    cluster_min_size,
    batch_size=128,
):
    print("assign_instance_or_add_cluster started.")

    batch_cnt = min(num_devices, len(stream_docs))
    # batches = [stream_docs[i::batch_cnt] for i in range(batch_cnt)]
    batches = [
        stream_docs[i : i + batch_size] for i in range(0, len(stream_docs), batch_size)
    ]
    lock = threading.Lock()

    def process_batch(batch, device):
        print(f"ㄴ Batch {len(batch)} starts on {device}.")

        # 1) 배치 임베딩 & 해시
        doc_ids = [doc["doc_id"] for doc in batch]
        batch_embs = [docs[doc_id]["TOKEN_EMBS"] for doc_id in doc_ids]
        batch_embs_tensor = torch.stack(batch_embs).to(device)  # (B, L, D)
        batch_hashes = lsh.encode_batch(
            batch_embeddings=batch_embs_tensor, is_sum=False
        )  # (B, hash_size, D)

        # 2) 배치-클러스터 유사도 계산 (벡터화)
        batch_closest_distances, batch_closest_ids = find_k_closest_clusters(
            batch_embs_tensor, clusters, 1, device, use_tensor_key, return_distance=True
        )  # (B, 1)

        # 3) 로컬에 할당 결과 수집
        local_assigns = defaultdict(list)
        local_new = []

        for i, doc in enumerate(batch):
            cid = batch_closest_ids[i][0]
            closest_distance = batch_closest_distances[i][0]
            closest_cluster = clusters[cid]
            closest_boundary = closest_cluster.get_boundary()
            n = len(closest_cluster.get_only_docids(docs, False))

            if n <= cluster_min_size or closest_distance <= closest_boundary:
                local_assigns[cid].append(
                    (doc_ids[i], batch_embs_tensor[i], batch_hashes[i])
                )
            else:
                local_new.append((batch_hashes[i], doc))

        # 4) 락 한 번만 걸고 글로벌 clusters 반영
        with lock:
            for cid, items in local_assigns.items():
                for doc_id, emb, hsh in items:
                    clusters[cid].assign(doc_id, emb, hsh, ts)
            for hsh, doc in local_new:
                clusters.append(Cluster(hsh, [doc], docs, use_tensor_key, ts))

    with ThreadPoolExecutor(max_workers=batch_cnt) as executor:
        futures = []
        for i, batch in enumerate(batches):
            device = devices[i % batch_cnt]
            futures.append(executor.submit(process_batch, batch, device))
        for future in futures:
            future.result()

    print(f"assign_instance_or_add_cluster finished.({len(clusters)})")
    return clusters


# def assign_instance_or_add_cluster(
#     lsh,
#     clusters,
#     stream_docs,
#     docs,
#     ts,
#     use_tensor_key,
#     cluster_min_size,
# ):
#     print("assign_instance_or_add_cluster started.")

#     batch_cnt = min(num_devices, len(stream_docs))
#     batches = [stream_docs[i::batch_cnt] for i in range(batch_cnt)]
#     lock = threading.Lock()

#     def process_batch(batch, device):
#         print(f"ㄴ Batch {len(batch)} starts on {device}.")

#         doc_ids = [doc["doc_id"] for doc in batch]
#         batch_embs = [docs[doc_id]["TOKEN_EMBS"] for doc_id in doc_ids]
#         batch_embs_tensor = torch.stack(batch_embs).to(device)  # (B, L, D)

#         # 배치 해시 인코딩
#         batch_hashes = lsh.encode_batch(
#             batch_embeddings=batch_embs_tensor, is_sum=False
#         )  # (B, hash_size, D)

#         # 배치 클러스터 유사도 계산
#         batch_closest_distances, batch_closest_ids = find_k_closest_clusters(
#             batch_embs_tensor, clusters, 1, device, lsh.use_tensor_key, return_distance=True
#         )  # (B, 1)

#         for i, doc in enumerate(batch):
#             doc_embs = batch_embs_tensor[i]
#             doc_hash = batch_hashes[i]

#             if len(clusters) == 0:
#                 with lock:
#                     clusters.append(Cluster(doc_hash, [doc], docs, use_tensor_key, ts))
#                 continue

#             closest_cluster_id = batch_closest_ids[i][0]
#             closest_cluster = clusters[closest_cluster_id]
#             # closest_distance = closest_cluster.get_distance(doc_embs)
#             closest_distance = batch_closest_distances[i][0]
#             # print(f"closest_cluster_id:{closest_cluster_id}, closest_distance: {closest_distance}")
#             closest_boundary = closest_cluster.get_boundary()
#             mean, std, n = (
#                 closest_cluster.calculate_mean(),
#                 closest_cluster.calculate_rms(),
#                 len(closest_cluster.get_only_docids(docs, False)),
#             )

#             if n <= cluster_min_size or closest_distance <= closest_boundary:
#                 with lock:
#                     closest_cluster.assign(doc_ids[i], doc_embs, doc_hash, ts)
#             else:
#                 with lock:
#                     # print(
#                     #     f"closest_cluster: {mean}, {std} | N: {n}, "
#                     #     f"doc_ids#:{len(closest_cluster.doc_ids)}, only docs#: {len(closest_cluster.get_only_docids(docs))}"
#                     # )
#                     print(
#                         f"closest_distance: {closest_distance}, closest_boundary: {closest_boundary}"
#                     )
#                     clusters.append(Cluster(doc_hash, [doc], docs, use_tensor_key, ts))

#     with ThreadPoolExecutor(max_workers=batch_cnt) as executor:
#         futures = []
#         for i, batch in enumerate(batches):
#             device = devices[i % batch_cnt]
#             futures.append(executor.submit(process_batch, batch, device))
#         for future in futures:
#             future.result()

#     print(f"assign_instance_or_add_cluster finished.({len(clusters)})")
#     return clusters


def get_samples_top_bottom_3(
    query,
    docs: dict,
    clusters,
    positive_k,
    negative_k,
    ts,
    use_tensor_key,
    verbose=True,
):
    closest_cluster_ids, farthest_cluster_ids = find_k_closest_clusters_for_sampling(
        token_embs=[query["TOKEN_EMBS"]],
        clusters=clusters,
        k=3,
        use_tensor_key=use_tensor_key,
    )
    cluster_ids = closest_cluster_ids[0]
    print(f"cluster_ids:{cluster_ids} ")
    first_cluster, second_cluster, third_cluster = (
        clusters[cluster_ids[0]],
        clusters[cluster_ids[1]],
        clusters[cluster_ids[2]],
    )
    (
        first_positive_samples,
        first_bottom_samples,
    ) = first_cluster.get_topk_docids_and_scores(query, docs, negative_k)
    (
        second_positive_samples,
        second_bottom_samples,
    ) = second_cluster.get_topk_docids_and_scores(query, docs, negative_k)
    (
        third_positive_samples,
        third_bottom_samples,
    ) = third_cluster.get_topk_docids_and_scores(query, docs, negative_k)
    combined_top_samples = sorted(
        first_positive_samples + second_positive_samples + third_positive_samples,
        key=lambda x: x[1],
        reverse=True,
    )
    combined_bottom_samples = sorted(
        first_bottom_samples + second_bottom_samples + third_bottom_samples,
        key=lambda x: x[1],
        reverse=True,
    )
    positive_samples = [x[0] for x in combined_top_samples[:positive_k]]
    negative_samples = [x[0] for x in combined_bottom_samples[-negative_k:]]
    if verbose:
        print(
            f" query: {query['doc_id']} | positive: {positive_samples} | negative:{negative_samples}"
        )
    return positive_samples, negative_samples


def get_samples_top_and_farthest3(
    query,
    docs: dict,
    clusters,
    positive_k,
    negative_k,
    ts,
    use_tensor_key,
    verbose=True,
):
    positive_samples, _ = get_samples_top_bottom_3(
        query, docs, clusters, positive_k, negative_k, ts, use_tensor_key, False
    )
    positive_sample = docs[positive_samples[0]]
    _, negative_samples = get_samples_top_bottom_3(
        positive_sample,
        docs,
        clusters,
        positive_k,
        negative_k,
        ts,
        use_tensor_key,
        False,
    )
    if verbose:
        print(
            f" query: {query['doc_id']} | positive: {positive_samples} | negative:{negative_samples}"
        )
    return positive_samples, negative_samples


def get_samples_top1_farthest_bottom6(
    query,
    docs: dict,
    clusters,
    positive_k,
    negative_k,
    ts,
    use_tensor_key,
    verbose=True,
):
    # Find top1 in 3 nearest clusters
    closest_cluster_ids, _ = find_k_closest_clusters_for_sampling(
        token_embs=[query["TOKEN_EMBS"]],
        clusters=clusters,
        k=3,
        use_tensor_key=use_tensor_key,
    )
    cluster_ids = closest_cluster_ids[0]
    print(f"cluster_ids:{cluster_ids} ")
    first_cluster, second_cluster, third_cluster = (
        clusters[cluster_ids[0]],
        clusters[cluster_ids[1]],
        clusters[cluster_ids[2]],
    )
    first_positive_samples, _ = first_cluster.get_topk_docids_and_scores(
        query, docs, negative_k
    )
    second_positive_samples, _ = second_cluster.get_topk_docids_and_scores(
        query, docs, negative_k
    )
    third_positive_samples, _ = third_cluster.get_topk_docids_and_scores(
        query, docs, negative_k
    )
    combined_top_samples = sorted(
        first_positive_samples + second_positive_samples + third_positive_samples,
        key=lambda x: x[1],
        reverse=True,
    )
    positive_samples = [x[0] for x in combined_top_samples[:positive_k]]

    # Find bottom6 in top1's the farthest cluster
    _, farthest_cluster_ids = find_k_closest_clusters_for_sampling(
        token_embs=[query["TOKEN_EMBS"]],
        clusters=clusters,
        k=1,
        use_tensor_key=use_tensor_key,
    )
    farthest_cluster = clusters[farthest_cluster_ids[0][0]]
    _, farthest_cluster_negative_samples = second_cluster.get_topk_docids_and_scores(
        docs[positive_samples[0]], docs, negative_k
    )
    negative_samples = [x[0] for x in farthest_cluster_negative_samples]

    if verbose:
        print(
            f" query: {query['doc_id']} | positive: {positive_samples} | negative:{negative_samples}"
        )
    return positive_samples, negative_samples


def get_samples_top_bottom_3_with_cache(
    caches,
    query,
    docs: dict,
    clusters,
    positive_k,
    negative_k,
    ts,
    use_tensor_key,
    verbose=True,
):
    closest_cluster_ids, farthest_cluster_ids = find_k_closest_clusters_for_sampling(
        token_embs=[query["TOKEN_EMBS"]],
        clusters=clusters,
        k=3,
        use_tensor_key=use_tensor_key,
    )
    cluster_ids = closest_cluster_ids[0]
    print(
        f"cluster_ids:{cluster_ids} / positive_k {positive_k}, negative_k {negative_k}"
    )
    first_cluster, second_cluster, third_cluster = (
        clusters[cluster_ids[0]],
        clusters[cluster_ids[1]],
        clusters[cluster_ids[2]],
    )
    (
        first_positive_samples,
        first_bottom_samples,
    ) = first_cluster.get_topk_docids_and_scores_with_cache(
        qid=query["doc_id"], cache=caches[cluster_ids[0]], docs=docs, k=negative_k
    )
    (
        second_positive_samples,
        second_bottom_samples,
    ) = second_cluster.get_topk_docids_and_scores_with_cache(
        qid=query["doc_id"], cache=caches[cluster_ids[1]], docs=docs, k=negative_k
    )
    (
        third_positive_samples,
        third_bottom_samples,
    ) = third_cluster.get_topk_docids_and_scores_with_cache(
        qid=query["doc_id"], cache=caches[cluster_ids[2]], docs=docs, k=negative_k
    )
    combined_top_samples = sorted(
        first_positive_samples + second_positive_samples + third_positive_samples,
        key=lambda x: x[1],
        reverse=True,
    )
    combined_bottom_samples = sorted(
        first_bottom_samples + second_bottom_samples + third_bottom_samples,
        key=lambda x: x[1],
        reverse=True,
    )
    positive_samples = [x[0] for x in combined_top_samples[:positive_k]]
    negative_samples = [x[0] for x in combined_bottom_samples[-negative_k:]]
    if verbose:
        print(
            f" query: {query['doc_id']} | positive: {positive_samples} | negative:{negative_samples}"
        )
    return positive_samples, negative_samples


def get_samples_top_and_farthest3_with_cache(
    caches,
    query,
    docs: dict,
    clusters,
    positive_k,
    negative_k,
    ts,
    use_tensor_key,
    verbose=True,
):
    positive_samples, _ = get_samples_top_bottom_3_with_cache(
        caches, query, docs, clusters, positive_k, negative_k, ts, use_tensor_key, False
    )
    positive_sample = docs[positive_samples[0]]
    # _, negative_samples = get_samples_top_bottom_3(
    _, negative_samples = get_samples_top_bottom_3_with_cache(
        caches,
        positive_sample,
        docs,
        clusters,
        positive_k,
        negative_k,
        ts,
        use_tensor_key,
        False,
    )
    if verbose:
        print(
            f" query: {query['doc_id']} | positive: {positive_samples} | negative:{negative_samples}"
        )
    return positive_samples, negative_samples


def evict_clusters(
    model, lsh, docs: dict, clusters: List[Cluster], ts, required_doc_size
) -> List[Cluster]:
    print("evict_cluster_instances started.")
    model.eval()
    remaining_clusters = []
    for cluster in clusters:
        is_updated = 1 if cluster.timestamp >= ts else 0
        is_alive = cluster.evict(model, lsh, docs, required_doc_size, is_updated)
        if is_alive:
            remaining_clusters.append(cluster)
    # stride. 순서 보장 필요X
    # cluster_chunks = [clusters[i::num_devices] for i in range(num_devices)]

    # def process_cluster_chunk(cluster_chunk, device):
    #     local_model = model.eval().to(device)
    #     local_result = []
    #     for cluster in cluster_chunk:
    #         is_updated = 1 if cluster.timestamp >= ts else 0
    #         is_alive = cluster.evict(
    #             local_model, lsh, docs, required_doc_size, is_updated
    #         )
    #         if is_alive:
    #             local_result.append(cluster)
    #     return local_result

    # remaining_clusters = []
    # with ThreadPoolExecutor(max_workers=num_devices) as executor:
    #     futures = {
    #         executor.submit(
    #             process_cluster_chunk, cluster_chunk, devices[i % num_devices]
    #         ): cluster_chunk
    #         for i, cluster_chunk in enumerate(cluster_chunks)
    #     }
    #     for future in futures:
    #         remaining_clusters.extend(future.result())
    # print(f"evict_cluster_instances finished. (left #{len(remaining_clusters)})")
    return remaining_clusters


def get_topk_docids(query, docs, doc_ids, k, batch_size=256) -> List[str]:
    query_token_embs = query["TOKEN_EMBS"].unsqueeze(0)

    def process_batch(device, batch_doc_ids):
        regl_scores = []
        temp_query_token_embs = query_token_embs.clone().to(device)
        for i in range(0, len(batch_doc_ids), batch_size):
            batch_ids = batch_doc_ids[i : i + batch_size]
            batch_emb = torch.stack(
                [docs[doc_id]["TOKEN_EMBS"] for doc_id in batch_ids], dim=0
            )
            if batch_emb.dim() > 3:
                batch_emb = batch_emb.squeeze()
            # E_q(batch_size, qlen, 768), E_d(batch_size, dlen, 768)
            # print(f'get_topk_docids | temp_query_token_embs:{temp_query_token_embs.shape}, batch_emb: {batch_emb.shape}')
            regl_score = calculate_S_qd_regl_batch(
                temp_query_token_embs, batch_emb, device
            )
            regl_scores.append(regl_score)
            # print(f"regl_scores: {len(regl_scores)}")
        regl_scores = torch.cat(regl_scores, dim=0)
        return [
            (doc_id, regl_scores[idx].item())
            for idx, doc_id in enumerate(batch_doc_ids)
        ]

    # stride, 순서 보장 필요X
    regl_scores = []
    batch_cnt = min(num_devices, len(doc_ids))
    doc_ids_batches = [doc_ids[i::batch_cnt] for i in range(batch_cnt)]
    with ThreadPoolExecutor() as executor:
        futures = []
        for i, device in enumerate(range(batch_cnt)):
            futures.append(executor.submit(process_batch, device, doc_ids_batches[i]))
        for future in futures:
            regl_scores.extend(future.result())

    combined_regl_scores = sorted(regl_scores, key=lambda x: x[1], reverse=True)
    top_k_regl_docs = combined_regl_scores[:k]
    top_k_regl_doc_ids = [x[0] for x in top_k_regl_docs]
    return top_k_regl_doc_ids


def retrieve_top_k_docs_from_cluster(
    stream, clusters, random_vectors, use_tensor_key, k
):
    print("retrieve_top_k_docs_from_cluster started.")
    # stride. 순서 보장 필요X
    doc_list = list(stream.docs.values())
    doc_batches = [
        doc_list[i::num_devices] for i in range(num_devices)
    ]  # only eval docs

    def doc_process_batch(batch, device, batch_size=256):
        print(f"ㄴ Document batch {len(batch)} starts on {device}.")
        cluster_docids_dict = defaultdict(list)
        mini_batches = [
            batch[i : i + batch_size] for i in range(0, len(batch), batch_size)
        ]
        for i, mini_batch in enumerate(mini_batches):
            print(f" ㄴ {i}th minibatch({(i+1)*batch_size}/{len(batch)})")
            mini_batch_embs = [doc["TOKEN_EMBS"] for doc in mini_batch]
            mini_batch_closest_ids = find_k_closest_clusters(
                mini_batch_embs, clusters, 1, device, use_tensor_key
            )
            for j in range(len(mini_batch_closest_ids)):
                closest_cluster_id = mini_batch_closest_ids[j][0]
                cluster_docids_dict[closest_cluster_id].append(mini_batch[j]["doc_id"])
        return cluster_docids_dict

    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        futures = {
            executor.submit(
                doc_process_batch, doc_batches[i], devices[i % num_devices]
            ): batch
            for i, batch in enumerate(doc_batches)
        }
    cluster_docids_dict = defaultdict(list)
    for future in futures:
        batch_result = future.result()
        for cluster_id, doc_ids in batch_result.items():
            cluster_docids_dict[cluster_id].extend(doc_ids)
    result = {}
    token_embs = [q["TOKEN_EMBS"] for q in stream.queries]
    batch_closest_ids = find_k_closest_clusters(
        token_embs, clusters, 3, devices[-1], use_tensor_key
    )
    for i, query in enumerate(stream.queries):
        print(f"Retrieval {i}th query.")
        # closest_cluster_id = batch_closest_ids[i][0]
        # cand_docids = cluster_docids_dict[closest_cluster_id]
        cand_docids = []
        for closest_cluster_id in batch_closest_ids[i]:
            cand_docids.extend(cluster_docids_dict[closest_cluster_id])
        print(f"ㄴ cand_docids {len(cand_docids)}")
        ans_ids = get_topk_docids(
            query=query, docs=stream.docs, doc_ids=cand_docids, k=k
        )
        result[query["doc_id"]] = ans_ids
    print("retrieve_top_k_docs_from_cluster finished.")
    return result


def clear_invalid_clusters(clusters: List[Cluster], docs: dict, required_doc_size):
    valid_clusters = []
    before_n = len(clusters)
    for cluster in clusters:
        if len(cluster.get_only_docids(docs)) >= required_doc_size:
            valid_clusters.append(cluster)
    after_n = len(valid_clusters)
    print(f"Clear invalid clusters #{before_n} -> #{after_n}")
    return valid_clusters


def clear_cluster_caches(clusters: List[Cluster]):
    for cluster in clusters:
        cluster.cache = {}
    return clusters


def deep_copy_tensor_dict(tensor_dict: dict) -> dict:
    return {k: v.clone() for k, v in tensor_dict.items()}


def compare_tensor_dicts(dict1: dict, dict2: dict) -> bool:
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if not torch.equal(dict1[key], dict2[key]):
            return False
    return True


def clear_unused_documents(clusters: List[Cluster], docs: dict, buffer_manager=None):
    print(f"clear_unused_documents | before total #{len(docs)}")
    all_used_doc_ids = set()
    for cluster in clusters:
        all_used_doc_ids.update(cluster.doc_ids)
    if buffer_manager:
        all_used_doc_ids.update(buffer_manager.get_all_ids())
    used_dict = {k: v for k, v in docs.items() if k in all_used_doc_ids}
    print(f"clear_unused_documents | after total #{len(used_dict)}")
    return used_dict


def make_query_psuedo_answers(queries, docs, clusters, use_tensor_key, k=1):
    print("make_query_psuedo_answers started.")
    result = {}
    for i, query in enumerate(queries):
        cluster_ids = find_k_closest_clusters_for_sampling(
            [query["TOKEN_EMBS"]], clusters, 1, use_tensor_key
        )[0]
        first_cluster = clusters[cluster_ids[0]]
        tops, _ = first_cluster.get_topk_docids(query, docs, 1)
        first_id = tops[0]
        result[query["doc_id"]] = first_id
        print(f"{i}th query is done.")
    print("make_query_psuedo_answers finished.")
    return result


def sample_past_queries(docs, clusters, n=100):
    print("sample_past_queries started.")
    all_qids = set()
    for cluster in clusters:
        all_qids.update(cluster.get_only_qids(docs))
    sampled_qids = random.sample(list(all_qids), min(n, len(all_qids)))
    sampled_queries = [v for k, v in docs.items() if k in sampled_qids]
    print(f"sample_past_queries finished.(sampled queries #{len(sampled_queries)})")
    return sampled_queries


def build_cluster_cache_table_by_query_result(
    qids,
    clusters,
    docs,
    query_result,
    query_batch_size=32,
    doc_batch_size=512,
):
    # --- Step 1: 쿼리 TOKEN_EMBS 캐시 (for regl) ---
    query_token_cache = {}
    for start in range(0, len(qids), query_batch_size):
        qid_batch = qids[start : start + query_batch_size]
        q_tensor = torch.cat(
            [docs[qid]["TOKEN_EMBS"].unsqueeze(0) for qid in qid_batch], dim=0
        )
        query_token_cache[tuple(qid_batch)] = q_tensor

    # --- Step 2: 클러스터 분할 ---
    cluster_splits = [[] for _ in devices]
    for idx, cluster in enumerate(clusters):
        cluster_splits[idx % len(devices)].append((idx, cluster))

    # --- Step 3: worker ---
    def worker(device, assigned_clusters):
        local_cache = {}
        for qid_batch_key, q_batch_tensor in query_token_cache.items():
            qid_batch = list(qid_batch_key)
            for cluster_id, cluster in assigned_clusters:
                cluster_doc_ids = set(cluster.get_only_docids(docs))
                k = min(100, len(cluster_doc_ids))
                for qid in qid_batch:
                    doc_embs = []
                    selected_topk_doc_ids = [
                        doc_id
                        for doc_id in query_result[qid]
                        if doc_id in cluster_doc_ids
                    ][:k]
                    if len(selected_topk_doc_ids) < k:
                        remaining_ids = list(
                            cluster_doc_ids - set(selected_topk_doc_ids)
                        )
                        random.shuffle(remaining_ids)
                        selected_topk_doc_ids += remaining_ids[
                            : k - len(selected_topk_doc_ids)
                        ]
                    selected_bottomk_doc_ids = [
                        doc_id
                        for doc_id in query_result[qid]
                        if doc_id in cluster_doc_ids
                    ][-k:]
                    if len(selected_bottomk_doc_ids) < k:
                        remaining_ids = list(
                            cluster_doc_ids - set(selected_bottomk_doc_ids)
                        )
                        random.shuffle(remaining_ids)
                        selected_bottomk_doc_ids += remaining_ids[
                            : k - len(selected_bottomk_doc_ids)
                        ]
                    selected_doc_ids = selected_topk_doc_ids + selected_bottomk_doc_ids
                    print(f"selected_doc_ids: {len(selected_doc_ids)}")
                    doc_embs.extend(
                        [
                            docs[doc_id]["TOKEN_EMBS"].unsqueeze(0)
                            for doc_id in selected_doc_ids
                        ]
                    )
                    d_batch = torch.cat(doc_embs, dim=0)
                    # print(f"q_batch_tensor:{q_batch_tensor.shape}, d_batch:{d_batch.shape}")
                    scores = calculate_S_qd_regl_batch_batch(
                        q_batch_tensor, d_batch, device
                    )  # (B_q, B_d)
                    doc_score_pairs = list(zip(selected_doc_ids, scores.tolist()))
                    sorted_pairs = sorted(
                        doc_score_pairs, key=lambda x: x[1], reverse=True
                    )
                    local_cache.setdefault(cluster_id, {})[qid] = sorted_pairs
        return local_cache

    # --- Step 6: 병렬 실행 ---
    all_local_caches = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(worker, devices[i], cluster_splits[i])
            for i in range(len(devices))
        ]
        for future in futures:
            all_local_caches.append(future.result())

    # --- Step 7: 결과 병합 ---
    cache = {}
    for local_cache in all_local_caches:
        for cluster_id, q_scores in local_cache.items():
            cache.setdefault(cluster_id, {}).update(q_scores)

    return cache


def build_cluster_cache_table_by_cosine(
    qids,
    clusters,
    docs,
    query_batch_size=32,
    doc_batch_size=512,
):
    """
    Builds a cache mapping each cluster and query to top-k/bottom-k similarity scores.
    Returns:
        cache: Dict[cluster_id][qid] = List[(doc_id, score)] sorted by score desc
    """

    # 1) 클러스터별 doc_ids와 doc_vecs([D, H])만 캐싱
    cluster_doc_id_cache = {}
    cluster_doc_vecs_cache = {}

    emb_dim = docs[qids[0]]["MEAN_EMB"].shape[-1]
    for cid, cluster in enumerate(clusters):
        doc_ids = cluster.get_only_docids(docs)
        cluster_doc_id_cache[cid] = doc_ids
        if doc_ids:
            cluster_doc_vecs_cache[cid] = torch.stack(
                [docs[doc_id]["MEAN_EMB"] for doc_id in doc_ids], dim=0
            )
        else:
            # 빈 클러스터 처리
            cluster_doc_vecs_cache[cid] = torch.empty(0, emb_dim)

    # 2) 쿼리 벡터만 batch 크기로 미리 묶어서 캐싱
    query_vec_batch_cache = []
    for i in range(0, len(qids), query_batch_size):
        qid_batch = qids[i : i + query_batch_size]
        q_vecs = torch.stack(
            [docs[qid]["MEAN_EMB"] for qid in qid_batch], dim=0
        )  # [B, H]
        query_vec_batch_cache.append((qid_batch, q_vecs))

    # 3) 클러스터-디바이스 분배
    cluster_splits = [[] for _ in devices]
    for idx, cluster in enumerate(clusters):
        cluster_splits[idx % len(devices)].append((idx, cluster))

    def worker(device, assigned_clusters):
        local_cache = {}

        for qid_batch, q_vecs in query_vec_batch_cache:
            q_vecs = q_vecs.to(device)  # [B, H]

            for cluster_id, _ in assigned_clusters:
                doc_ids = cluster_doc_id_cache[cluster_id]
                doc_vecs = cluster_doc_vecs_cache[cluster_id].to(device)  # [D, H]
                sim_matrix = torch.matmul(q_vecs, doc_vecs.T)  # [B, D]
                # q_cnt = (
                #     int(len(cluster_doc_ids) * (len(cluster_qids) / len(cluster_qids)))
                #     * 3
                #     if len(cluster_qids) != 0
                #     else 50
                # )
                # k = min(max(q_cnt, 50), len(valid_ids))
                k = min(50, len(doc_ids))
                topk_idx = torch.topk(sim_matrix, k=k, dim=1, largest=True).indices
                bottomk_idx = torch.topk(sim_matrix, k=k, dim=1, largest=False).indices

                for i, qid in enumerate(qid_batch):
                    idxs = torch.cat([topk_idx[i], bottomk_idx[i]]).tolist()
                    sel_doc_ids = [doc_ids[j] for j in idxs]
                    q_tensor = (
                        docs[qid]["TOKEN_EMBS"].unsqueeze(0).to(device)
                    )  # [1, T, H]
                    d_batch = torch.stack(
                        [docs[d]["TOKEN_EMBS"] for d in sel_doc_ids], dim=0
                    ).to(
                        device
                    )  # [2k, T, H]

                    scores = calculate_S_qd_regl_batch_batch(
                        q_tensor, d_batch, device
                    )  # [1, 2k]
                    pairs = list(zip(sel_doc_ids, scores.tolist()[0]))
                    local_cache.setdefault(cluster_id, {})[qid] = sorted(
                        pairs, key=lambda x: x[1], reverse=True
                    )

        return local_cache

    from concurrent.futures import ThreadPoolExecutor

    all_caches = []
    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        futures = [
            ex.submit(worker, devices[i], cluster_splits[i])
            for i in range(len(devices))
        ]
        for f in futures:
            all_caches.append(f.result())

    # 4) 합치기
    cache = {}
    for lc in all_caches:
        for cid, q_scores in lc.items():
            cache.setdefault(cid, {}).update(q_scores)

    return cache


# def build_cluster_cache_table(
#     qids, clusters: List, docs: dict, query_batch_size=4, doc_batch_size=512
# ):
#     """
#     {
#         cluster_id: {
#             qid: [cluster_docs_len]
#         }
#     }
#     """
#     cluster_splits = [[] for _ in devices]
#     for idx, cluster in enumerate(clusters):
#         cluster_splits[idx % len(devices)].append((idx, cluster))

#     query_tensor_cache = {}
#     for q_start in range(0, len(qids), query_batch_size):
#         qid_batch = qids[q_start : q_start + query_batch_size]
#         qid_key = tuple(qid_batch)
#         if qid_key not in query_tensor_cache:
#             q_embs = [docs[qid]["TOKEN_EMBS"].unsqueeze(0) for qid in qid_batch]
#             query_tensor_cache[qid_key] = torch.cat(q_embs, dim=0)

#     def worker(device, assigned_clusters):
#         local_cache = {}
#         doc_tensor_cache = {}
#         qidx = 0
#         for qid_key, q_batch_tensor in query_tensor_cache.items():
#             qid_batch = list(qid_key)
#             print(f"device {device} | {qidx}th query batch")
#             qidx += 1
#             for cluster_id, cluster in assigned_clusters:
#                 if cluster_id not in doc_tensor_cache:
#                     doc_ids = cluster.get_only_docids(docs)
#                     doc_embs = [
#                         docs[doc_id]["TOKEN_EMBS"].unsqueeze(0)
#                         for doc_id in doc_ids
#                         if doc_id in docs
#                     ]
#                     doc_tensor_cache[cluster_id] = doc_embs
#                 else:
#                     doc_embs = doc_tensor_cache[cluster_id]

#                 if not doc_embs:
#                     for qid in qid_batch:
#                         local_cache.setdefault(cluster_id, {})[qid] = 0
#                     continue

#                 all_scores = []
#                 didx = 0
#                 for d_start in range(0, len(doc_embs), doc_batch_size):
#                     print(f"ㄴ {cluster_id} {didx}th documents batch")
#                     d_batch = torch.cat(
#                         doc_embs[d_start : d_start + doc_batch_size], dim=0
#                     )
#                     scores = calculate_S_qd_regl_batch_batch(
#                         q_batch_tensor, d_batch, device
#                     )
#                     all_scores.append(scores)
#                     didx += 1

#                 full_score_matrix = torch.cat(all_scores, dim=1)
#                 for i, qid in enumerate(qid_batch):
#                     local_cache.setdefault(cluster_id, {})[qid] = full_score_matrix[
#                         i
#                     ].tolist()

#         return local_cache

#     all_local_caches = []
#     with ThreadPoolExecutor(max_workers=len(devices)) as executor:
#         futures = [
#             executor.submit(worker, device, cluster_splits[i])
#             for i, device in enumerate(devices)
#         ]
#         for future in futures:
#             all_local_caches.append(future.result())

#     cluster_cache = {}
#     for local in all_local_caches:
#         for cluster_id, qid_dict in local.items():
#             cluster_cache.setdefault(cluster_id, {}).update(qid_dict)

#     return cluster_cache
