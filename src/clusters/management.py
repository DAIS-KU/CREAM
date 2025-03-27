import copy
from concurrent.futures import ThreadPoolExecutor
from typing import List
from collections import defaultdict
import torch
import random

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
    model, stream_docs, docs, k, nbits, lsh, max_iters, use_tensor_key=True
) -> List[Cluster]:
    _, enoded_stream_docs = renew_data(
        queries=None,
        documents=stream_docs,
        renew_q=False,
        renew_d=True,
        nbits=nbits,
        use_tensor_key=use_tensor_key,
        model_path="../data/base_model.pth",
    )
    enoded_stream_docs = list(enoded_stream_docs.values())
    if use_tensor_key:
        centroids, cluster_instances = kmeans_pp_use_tensor_key(
            enoded_stream_docs, k, max_iters, nbits
        )
    else:
        centroids, cluster_instances = kmeans_pp(
            enoded_stream_docs, k, max_iters, nbits
        )
    clusters = []
    for cid, centroid in enumerate(centroids):
        if len(cluster_instances[cid]):
            print(f"Create {len(clusters)}th Cluster.")
            clusters.append(
                Cluster(
                    model, lsh, centroid, cluster_instances[cid], docs, use_tensor_key
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
            batch_prototypes = torch.stack(prototypes[i : i + batch_size]).to(device)
            batch_scores = calculate_S_qd_regl_batch_batch(
                token_embs, batch_prototypes, device
            ).cpu()  # (t_bsz, p_bsz)
            scores.append(batch_scores)
    else:
        for prototype in prototypes:
            score = calculate_S_qd_regl_dict(token_embs, prototype, device)
            scores.append(score.unsqueeze(1))
    scores_tensor = torch.cat(scores, dim=1)  # (t_bsz, len(prototypes))
    topk_values, topk_indices = torch.topk(scores_tensor, k, dim=1)  # 각 샘플별 k개 선택
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
                batch_tensor = torch.stack(batch_prototypes[i : i + batch_size]).to(
                    device
                )
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
    else:
        for prototype in prototypes:
            score = calculate_S_qd_regl_dict(token_embs, prototype, devices[1])
            scores.append(score.unsqueeze(1))
    scores_tensor = torch.cat(scores, dim=1)
    topk_values, topk_indices = torch.topk(scores_tensor, k, dim=1)  # 각 샘플별 k개 선택
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
        for i, doc in enumerate(batch):
            doc_embs = batch_embs[i]
            doc_hash = lsh.encode(doc_embs)
            if len(clusters) == 0:
                print(f"Empty Clusters")
                clusters.append(
                    Cluster(temp_model, lsh, doc_hash, [doc], docs, use_tensor_key, ts)
                )
            else:
                closest_cluster_id = batch_closest_ids[i][0]
                closest_cluster = clusters[closest_cluster_id]
                s1, s2, n = closest_cluster.get_statistics()
                closest_distance = closest_cluster.get_distance(doc_embs)
                closest_boundary = closest_cluster.get_boundary()
                if n <= cluster_min_size or closest_distance <= closest_boundary:
                    closest_cluster.assign(doc_ids[i], doc_embs, doc_hash, ts)
                else:
                    print(f"closest_cluster: {s1}, {s2}, {n}")
                    print(
                        f"closest_distance: {closest_distance}, closest_boundary:{closest_boundary}"
                    )
                    clusters.append(
                        Cluster(
                            temp_model, lsh, doc_hash, [doc], docs, use_tensor_key, ts
                        )
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
    model,
    query,
    docs: dict,
    clusters,
    positive_k,
    negative_k,
    ts,
    use_tensor_key,
    candidate_num=5,
):
    cluster_ids = find_k_closest_clusters_for_sampling(
        model=model,
        texts=[query["query"]],
        clusters=clusters,
        k=2,
        use_tensor_key=use_tensor_key,
    )[0]
    print(f"cluster_ids:{cluster_ids} ")
    first_cluster, second_cluster = clusters[cluster_ids[0]], clusters[cluster_ids[1]]

    first_samples = first_cluster.get_topk_docids_and_scores(
        model, query, docs, candidate_num
    )
    second_samples = second_cluster.get_topk_docids_and_scores(
        model, query, docs, candidate_num
    )
    combined_samples = sorted(
        first_samples + second_samples, key=lambda x: x[1], reverse=True
    )
    top_k_doc_ids = [x[0] for x in combined_samples]

    positive_samples, negative_samples = (
        top_k_doc_ids[:positive_k],
        top_k_doc_ids[positive_k : positive_k + negative_k],
    )
    print(
        f" query: {query['qid']} | positive: {positive_samples} | negative:{negative_samples}"
    )
    return positive_samples, negative_samples, []


def evict_clusters(
    model, lsh, docs: dict, clusters: List[Cluster], ts, required_doc_size
) -> List[Cluster]:
    print("evict_cluster_instances started.")
    # stride. 순서 보장 필요X
    cluster_chunks = [clusters[i::num_devices] for i in range(num_devices)]

    def process_cluster_chunk(cluster_chunk, device):
        local_model = copy.deepcopy(model).to(device)
        local_result = []
        for cluster in cluster_chunk:
            if cluster.timestamp >= ts:
                # cluster.refresh_prototype_with_cache()
                is_alive = True
            else:
                is_alive = cluster.evict(local_model, lsh, docs, required_doc_size)
            if is_alive:
                cluster.decay()
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
    query_token_embs = get_passage_embeddings(model, query["query"], devices[-1])

    def process_batch(device, batch_doc_ids):
        regl_scores = []
        temp_model = copy.deepcopy(model).to(device)
        temp_query_token_embs = query_token_embs.clone().to(device)
        for i in range(0, len(batch_doc_ids), batch_size):
            batch_ids = batch_doc_ids[i : i + batch_size]
            batch_texts = [docs[doc_id]["text"] for doc_id in batch_ids]
            batch_emb = get_passage_embeddings(temp_model, batch_texts, device)
            if batch_emb.dim() > 3:
                batch_emb = batch_emb.squeeze()
            # print(f"get_topk_docids({device}) | batch_emb:{batch_emb.shape}")
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
    model, stream, clusters, random_vectors, use_tensor_key, k
):
    print("retrieve_top_k_docs_from_cluster started.")
    # stride. 순서 보장 필요X
    doc_list = list(stream.docs.values())
    doc_batches = [
        doc_list[i::num_devices] for i in range(num_devices)
    ]  # only eval docs

    def doc_process_batch(batch, device, batch_size=128):
        print(f"ㄴ Document batch {len(batch)} starts on {device}.")
        cluster_docids_dict = defaultdict(list)
        temp_model = copy.deepcopy(model).to(device)
        mini_batches = [
            batch[i : i + batch_size] for i in range(0, len(batch), batch_size)
        ]
        for i, mini_batch in enumerate(mini_batches):
            print(f" ㄴ {i}th minibatch({(i+1)*batch_size}/{len(batch)})")
            mini_batch_texts = [doc["text"] for doc in mini_batch]
            mini_batch_closest_ids = find_k_closest_clusters(
                temp_model, mini_batch_texts, clusters, 1, device, use_tensor_key
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
    texts = [q["query"] for q in stream.queries]
    batch_closest_ids = find_k_closest_clusters(
        model, texts, clusters, 1, devices[1], use_tensor_key
    )
    for i, query in enumerate(stream.queries):
        print(f"Retrieval {i}th query.")
        closest_cluster_id = batch_closest_ids[i][0]
        cand_docids = cluster_docids_dict[closest_cluster_id]
        print(f"ㄴ cand_docids {len(cand_docids)}")
        ans_ids = get_topk_docids(
            model=model, query=query, docs=stream.docs, doc_ids=cand_docids, k=k
        )
        result[query["qid"]] = ans_ids
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


def make_query_psuedo_answers(model, queries, docs, clusters, use_tensor_key, k=1):
    print("make_query_psuedo_answers started.")
    result = {}
    for i, query in enumerate(queries):
        cluster_ids = find_k_closest_clusters_for_sampling(
            model, [query["query"]], clusters, 1, use_tensor_key
        )[0]
        first_cluster = clusters[cluster_ids[0]]
        first_id = first_cluster.get_topk_docids(model, query, docs, 1)[0]
        result[query["qid"]] = first_id
        print(f"{i}th query is done.")
    print("make_query_psuedo_answers finished.")
    return result
