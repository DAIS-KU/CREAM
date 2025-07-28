from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import List, Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from functions import (
    encode_texts_mean_pooling,
    get_top_k_documents_cosine,
    process_batch,
)

from .cluster import Cluster
from .clustering import kmeans_mean_pooling

tokenizer = BertTokenizer.from_pretrained(
    "/home/work/retrieval/bert-base-uncased/bert-base-uncased"
)
MAX_SCORE = 1.0
num_devices = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_devices)]


def initialize(
    model, stream_docs, docs, k, max_iters, use_tensor_key=True
) -> List[Cluster]:
    enoded_stream_docs = encode_cluster_data_mean_pooling(
        documents_data=stream_docs,
        model=model,
    )
    enoded_stream_docs = list(enoded_stream_docs.values())
    centroids, cluster_instances = kmeans_mean_pooling(enoded_stream_docs, k, max_iters)

    clusters = []
    for cid, centroid in enumerate(centroids):
        if len(cluster_instances[cid]):
            print(f"Create {len(clusters)}th Cluster.")
            clusters.append(
                Cluster(model, centroid, cluster_instances[cid], docs, use_tensor_key)
            )
    return clusters


def find_k_closest_clusters(
    model,
    embs: List[Any],
    clusters: List[Cluster],
    k,
    device,
    use_tensor_key,
    batch_size=8,
) -> List[int]:
    prototypes = [cluster.prototype for cluster in clusters]  # (num_clusters, dim)
    # print(f"find_k_closest_clusters | len(prototypes): {len(prototypes)}")
    scores = []
    for i in range(0, len(prototypes), batch_size):
        batch = prototypes[i : i + batch_size]
        batch = [t.to(device) for t in batch]
        batch_prototypes = torch.stack(batch, dim=0)  # (batch_size, dim)

        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        batch_prototypes = torch.nn.functional.normalize(batch_prototypes, p=2, dim=-1)
        # print(f"find_k_closest_clusters | embs:{embs.shape}, batch_prototypes:{batch_prototypes.shape}")
        batch_scores = F.cosine_similarity(
            embs.unsqueeze(1),  # (num_samples, 1, dim)
            batch_prototypes.unsqueeze(0).to(embs.device),  # (1, batch_size, dim)
            dim=2,
        )  # (num_samples, batch_size)
        scores.append(batch_scores)
    scores_tensor = torch.cat(scores, dim=1)  # (num_samples, len(prototypes))
    topk_values, topk_indices = torch.topk(scores_tensor, k, dim=1)
    return topk_indices.tolist()


def find_k_closest_clusters_for_sampling(
    embs,
    clusters: List[Cluster],
    k,
    batch_size=8,
) -> List[int]:
    prototypes = [
        cluster.prototype.cpu() for cluster in clusters
    ]  # (num_clusters, dim)
    num_workers = min(len(prototypes), num_devices)
    q, r = divmod(len(prototypes), num_workers)
    prototype_batches = [
        prototypes[i * q : (i + 1) * q] for i in range(num_workers - 1)
    ] + [prototypes[(num_workers - 1) * q :]]
    scores = []

    def process_on_device(device, batch_prototypes):
        device_scores = []
        temp_token_embs = embs.clone().to(device)
        for i in range(0, len(batch_prototypes), batch_size):
            batch_tensor = torch.stack(
                batch_prototypes[i : i + batch_size], dim=0
            )  # (batch_size, dim)
            temp_token_embs = torch.nn.functional.normalize(
                temp_token_embs, p=2, dim=-1
            )
            batch_tensor = torch.nn.functional.normalize(batch_tensor, p=2, dim=-1)
            # print(f"find_k_closest_clusters_for_sampling | temp_token_embs;{temp_token_embs.shape}, batch_tensor:{batch_tensor.shape}")
            batch_scores = F.cosine_similarity(
                temp_token_embs.to(device), batch_tensor.to(device), dim=1
            ).cpu()  # (batch_size, num_clusters)
            if batch_scores.dim() == 1:
                batch_scores = batch_scores.unsqueeze(0)
            device_scores.append(batch_scores)
        # print(f"find_k_closest_clusters_for_sampling-process_on_device device_scores{len(device_scores)}/{device_scores[0].shape}")
        res = torch.cat(device_scores, dim=1)
        return res

    with ThreadPoolExecutor(num_workers) as executor:
        scores = list(executor.map(process_on_device, devices, prototype_batches))

    scores_tensor = torch.cat(scores, dim=1)  # (num_samples, num_clusters)
    topk_values, topk_indices = torch.topk(scores_tensor, k, dim=1)
    bottomk_values, bottomk_indices = torch.topk(scores_tensor, k, dim=1, largest=False)
    return topk_indices.tolist(), bottomk_indices.tolist()


def get_samples_top_bottom(
    model,
    query,
    docs: dict,
    clusters,
    positive_k,
    negative_k,
):
    closest_cluster_ids, farthest_cluster_ids = find_k_closest_clusters_for_sampling(
        embs=query["EMB"],
        clusters=clusters,
        k=3,
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
    ) = first_cluster.get_topk_docids_and_scores(model, query, docs, negative_k)
    (
        second_positive_samples,
        second_bottom_samples,
    ) = second_cluster.get_topk_docids_and_scores(model, query, docs, negative_k)
    (
        third_positive_samples,
        third_bottom_samples,
    ) = third_cluster.get_topk_docids_and_scores(model, query, docs, negative_k)
    combined_samples = sorted(
        first_positive_samples
        + first_bottom_samples
        + second_positive_samples
        + second_bottom_samples
        + third_positive_samples
        + third_bottom_samples,
        key=lambda x: x[1],
        reverse=True,
    )
    top_k_doc_ids = [x[0] for x in combined_samples]
    positive_samples, negative_samples = (
        top_k_doc_ids[:1],
        top_k_doc_ids[-6:],
    )

    print(
        f" query: {query['doc_id']} | positive: {positive_samples} | negative:{negative_samples}"
    )
    return positive_samples, negative_samples


def assign_instance_or_add_cluster(
    model,
    clusters: List[Cluster],
    stream_docs,
    docs: dict,
    ts,
    use_tensor_key,
    cluster_min_size,
):
    print("assign_instance_or_add_cluster started.")

    batch_cnt = min(num_devices, len(stream_docs))
    batches = [stream_docs[i::batch_cnt] for i in range(batch_cnt)]

    def process_batch(batch, device):
        print(f"ㄴ Batch {len(batch)} starts on {device}.")
        temp_model = deepcopy(model).to(device)
        doc_ids = [doc["doc_id"] for doc in batch]
        batch_embs = torch.stack([doc["EMB"] for doc in batch], dim=0)
        # print(f"assign_instance_or_add_cluster | batch_embs: {batch_embs.shape}")
        batch_closest_ids = find_k_closest_clusters(
            temp_model, batch_embs, clusters, 1, device, use_tensor_key
        )
        for i, doc in enumerate(batch):
            doc_embs = batch_embs[i]
            if len(clusters) == 0:
                print(f"Empty Clusters")
                clusters.append(
                    Cluster(temp_model, doc_embs, [doc], docs, use_tensor_key, ts)
                )
            else:
                closest_cluster_id = batch_closest_ids[i][0]
                closest_cluster = clusters[closest_cluster_id]
                s1, s2, n = closest_cluster.get_statistics()
                closest_distance = closest_cluster.get_distance(doc_embs)
                closest_boundary = closest_cluster.get_boundary()
                if n <= cluster_min_size or closest_distance <= closest_boundary:
                    closest_cluster.assign(doc_ids[i], doc_embs, ts)
                else:
                    print(f"closest_cluster: {s1}, {s2}, {n}")
                    print(
                        f"closest_distance: {closest_distance}, closest_boundary:{closest_boundary}"
                    )
                    clusters.append(
                        Cluster(temp_model, doc_embs, [doc], docs, use_tensor_key, ts)
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


def evict_clusters(
    model, docs: dict, clusters: List[Cluster], ts, required_doc_size
) -> List[Cluster]:
    print("evict_cluster_instances started.")
    # stride. 순서 보장 필요X
    cluster_chunks = [clusters[i::num_devices] for i in range(num_devices)]

    def process_cluster_chunk(cluster_chunk, device):
        local_model = deepcopy(model).to(device)
        local_result = []
        for cluster in cluster_chunk:
            is_updated = 1 if cluster.timestamp >= ts else 0
            is_alive = cluster.evict(local_model, docs, required_doc_size, is_updated)
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


def get_topk_docids(model, query, docs, doc_ids, k, batch_size=128) -> List[str]:
    query_token_embs = query["EMB"]

    def process_batch(device, batch_doc_ids):
        regl_scores = []
        temp_model = deepcopy(model).to(device)
        temp_query_token_embs = query_token_embs.clone().to(device)
        for i in range(0, len(batch_doc_ids), batch_size):
            batch_ids = batch_doc_ids[i : i + batch_size]
            batch_emb = torch.stack(
                [docs[doc_id]["EMB"] for doc_id in batch_ids], dim=1
            )

            if batch_emb.dim() > 3:
                batch_emb = batch_emb.squeeze()
            batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=-1)
            temp_query_token_embs = torch.nn.functional.normalize(
                temp_query_token_embs, p=2, dim=-1
            )
            cosine_sim = F.cosine_similarity(
                batch_emb, temp_query_token_embs.unsqueeze(0), dim=-1
            ).squeeze(0)
            distance = 1 - cosine_sim
            regl_scores.append(distance)
        regl_scores = torch.cat(regl_scores, dim=0)
        return [
            (doc_id, regl_scores[idx].item())
            for idx, doc_id in enumerate(batch_doc_ids)
        ]

    regl_scores = []
    batch_cnt = min(num_devices, len(doc_ids))
    doc_ids_batches = [doc_ids[i::batch_cnt] for i in range(batch_cnt)]

    with ThreadPoolExecutor() as executor:
        futures = []
        for i, device in enumerate(range(batch_cnt)):
            futures.append(executor.submit(process_batch, device, doc_ids_batches[i]))

        for future in futures:
            regl_scores.extend(future.result())

    sorted_regl_scores = sorted(regl_scores, key=lambda x: x[1])
    top_k_regl_doc_ids = [x[0] for x in sorted_regl_scores[:k]]
    return top_k_regl_doc_ids


def retrieve_top_k_docs_from_cluster(model, stream, clusters, use_tensor_key, k):
    print("retrieve_top_k_docs_from_cluster started.")
    doc_list = list(stream.docs.values())
    doc_batches = [
        doc_list[i::num_devices] for i in range(num_devices)
    ]  # only eval docs

    def doc_process_batch(batch, device, batch_size=128):
        print(f"ㄴ Document batch {len(batch)} starts on {device}.")
        cluster_docids_dict = defaultdict(list)
        temp_model = deepcopy(model).to(device)
        mini_batches = [
            batch[i : i + batch_size] for i in range(0, len(batch), batch_size)
        ]
        for i, mini_batch in enumerate(mini_batches):
            print(f" ㄴ {i}th minibatch({(i+1)*batch_size}/{len(batch)})")
            mini_batch_embs = torch.stack([doc["EMB"] for doc in mini_batch], dim=1)
            mini_batch_closest_ids = find_k_closest_clusters(
                temp_model, mini_batch_embs, clusters, 1, device, use_tensor_key
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
    embs = torch.stack([q["EMB"] for q in stream.queries], dim=1)
    batch_closest_ids = find_k_closest_clusters(
        model, embs, clusters, 1, devices[-1], use_tensor_key
    )
    for i, query in enumerate(stream.queries):
        print(f"Retrieval {i}th query.")
        closest_cluster_id = batch_closest_ids[i][0]
        cand_docids = cluster_docids_dict[closest_cluster_id]
        print(f"ㄴ cand_docids {len(cand_docids)}")
        ans_ids = get_topk_docids(
            model=model, query=query, docs=stream.docs, doc_ids=cand_docids, k=k
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


def encode_queries_mean_pooling(model, queries_data):
    num_gpus = torch.cuda.device_count()
    models = [deepcopy(model) for _ in range(num_gpus)]
    query_batches = []

    for i in range(num_gpus):
        query_start = i * len(queries_data) // num_gpus
        query_end = (i + 1) * len(queries_data) // num_gpus
        query_batches.append(queries_data[query_start:query_end])

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        print(f"Query mean-pooling embedding starts.")
        futures = []
        for i in range(num_gpus):
            if query_batches[i]:
                futures.append(
                    executor.submit(
                        process_batch,
                        query_batches[i],
                        models[i],
                        tokenizer,
                        i,
                        "qid",
                        "query",
                    )
                )
        query_results = {}
        for future in futures:
            query_results.update(future.result())
    return query_results


def encode_cluster_data_mean_pooling(model, documents_data):
    num_gpus = torch.cuda.device_count()
    models = [deepcopy(model) for _ in range(num_gpus)]
    doc_batches = []

    for i in range(num_gpus):
        doc_start = i * len(documents_data) // num_gpus
        doc_end = (i + 1) * len(documents_data) // num_gpus
        doc_batches.append(documents_data[doc_start:doc_end])

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        print(f"Document mean-pooling embedding starts..")
        futures = []
        for i in range(num_gpus):
            if doc_batches[i]:
                futures.append(
                    executor.submit(
                        process_batch,
                        doc_batches[i],
                        models[i],
                        tokenizer,
                        i,
                        "doc_id",
                        "text",
                    )
                )
        doc_results = {}
        for future in futures:
            doc_results.update(future.result())
    return doc_results


def make_cos_query_psuedo_answers(model, queries, docs, clusters, k=1):
    print("make_query_psuedo_answers started.")
    num_gpus = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]

    result = {}
    for i, query in enumerate(queries):
        cluster_ids = find_k_closest_clusters_for_sampling(
            torch.stack([query["EMB"]], dim=0), clusters, 1
        )[0]
        first_cluster = clusters[cluster_ids[0]]
        result[query["doc_id"]] = first_cluster.get_topk_docids(model, query, docs, k)[
            0
        ]
        print(f"{i}th query is done.")
    print("make_query_psuedo_answers finished.")
    return result


def clear_unused_documents(clusters: List[Cluster], docs: dict):
    print(f"clear_unused_documents | before total #{len(docs)}")
    all_used_doc_ids = set()
    for cluster in clusters:
        all_used_doc_ids.update(cluster.doc_ids)
    used_dict = {k: v for k, v in docs.items() if k in all_used_doc_ids}
    print(f"clear_unused_documents | after total #{len(used_dict)}")
    return used_dict


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

    emb_dim = docs[qids[0]]["EMB"].shape[-1]
    for cid, cluster in enumerate(clusters):
        doc_ids = cluster.get_only_docids(docs)
        cluster_doc_id_cache[cid] = doc_ids
        if doc_ids:
            cluster_doc_vecs_cache[cid] = torch.stack(
                [docs[doc_id]["EMB"] for doc_id in doc_ids], dim=0
            )
        else:
            # 빈 클러스터 처리
            cluster_doc_vecs_cache[cid] = torch.empty(0, emb_dim)

    # 2) 쿼리 벡터만 batch 크기로 미리 묶어서 캐싱
    query_vec_batch_cache = []
    for i in range(0, len(qids), query_batch_size):
        qid_batch = qids[i : i + query_batch_size]
        q_vecs = torch.stack([docs[qid]["EMB"] for qid in qid_batch], dim=0)  # [B, H]
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
                k = min(50, len(doc_ids))
                topk_idx = torch.topk(sim_matrix, k=k, dim=1, largest=True).indices
                bottomk_idx = torch.topk(sim_matrix, k=k, dim=1, largest=False).indices

                for i, qid in enumerate(qid_batch):
                    idxs = torch.cat([topk_idx[i], bottomk_idx[i]]).tolist()
                    sel_doc_ids = [doc_ids[j] for j in idxs]
                    sel_sims = sim_matrix[i, idxs].tolist()
                    pairs = list(zip(sel_doc_ids, sel_sims))
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
        embs=query["EMB"],
        clusters=clusters,
        k=3,
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
