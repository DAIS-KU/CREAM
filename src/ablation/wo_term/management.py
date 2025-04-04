from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import List

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

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
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
    texts: List[str],
    clusters: List[Cluster],
    k,
    device,
    use_tensor_key,
    batch_size=8,
) -> List[int]:
    token_embs = encode_texts_mean_pooling(model, texts)  # (num_samples, dim)
    prototypes = [cluster.prototype for cluster in clusters]  # (num_clusters, dim)
    scores = []
    for i in range(0, len(prototypes), batch_size):
        batch = prototypes[i : i + batch_size]
        batch = [t.to(device) for t in batch]
        batch_prototypes = torch.stack(batch)  # (batch_size, dim)
        batch_scores = F.cosine_similarity(
            token_embs.unsqueeze(1),  # (num_samples, 1, dim)
            batch_prototypes.unsqueeze(0).to(token_embs),  # (1, batch_size, dim)
            dim=2,
        )  # (num_samples, batch_size)
        scores.append(batch_scores)
    scores_tensor = torch.cat(scores, dim=1)  # (num_samples, len(prototypes))
    topk_values, topk_indices = torch.topk(scores_tensor, k, dim=1)
    return topk_indices.tolist()


def find_k_closest_clusters_for_sampling(
    model,
    texts: List[str],
    clusters: List[Cluster],
    k,
    batch_size=8,
) -> List[int]:
    token_embs = encode_texts_mean_pooling(model, texts)  # (num_samples, dim)
    prototypes = [cluster.prototype for cluster in clusters]  # (num_clusters, dim)
    scores = []
    prototypes_cpu = [p.cpu() for p in prototypes]
    prototype_batches = list(
        map(
            lambda batch: [torch.tensor(proto) for proto in batch],
            np.array_split(prototypes_cpu, num_devices),
        )
    )

    def process_on_device(device, batch_prototypes):
        device_scores = []
        temp_token_embs = token_embs.clone().to(device)
        for i in range(0, len(batch_prototypes), batch_size):
            batch_tensor = torch.stack(batch_prototypes[i : i + batch_size]).to(
                device
            )  # (batch_size, dim)
            batch_scores = F.cosine_similarity(
                temp_token_embs, batch_tensor, dim=1
            ).cpu()  # (batch_size, num_clusters)
            if batch_scores.dim() == 1:
                batch_scores = batch_scores.unsqueeze(0)
            device_scores.append(batch_scores)
        # print(f"find_k_closest_clusters_for_sampling-process_on_device device_scores{len(device_scores)}/{device_scores[0].shape}")
        res = torch.cat(device_scores, dim=1)
        return res

    with ThreadPoolExecutor(num_devices) as executor:
        scores = list(executor.map(process_on_device, devices, prototype_batches))

    scores_tensor = torch.cat(scores, dim=1)  # (num_samples, num_clusters)
    topk_values, topk_indices = torch.topk(scores_tensor, k, dim=1)
    return topk_indices.tolist()


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
        texts = [doc["text"] for doc in batch]
        doc_ids = [doc["doc_id"] for doc in batch]
        batch_embs = encode_texts_mean_pooling(temp_model, texts).squeeze()
        batch_closest_ids = find_k_closest_clusters(
            temp_model, texts, clusters, 1, device, use_tensor_key
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
            if cluster.timestamp >= ts:
                # cluster.refresh_prototype_with_cache()
                is_alive = True
            else:
                is_alive = cluster.evict(local_model, docs, required_doc_size)
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
    query_token_embs = encode_texts_mean_pooling(model, query["query"])

    def process_batch(device, batch_doc_ids):
        regl_scores = []
        temp_model = deepcopy(model).to(device)
        temp_query_token_embs = query_token_embs.clone().to(device)
        for i in range(0, len(batch_doc_ids), batch_size):
            batch_ids = batch_doc_ids[i : i + batch_size]
            batch_texts = [docs[doc_id]["text"] for doc_id in batch_ids]
            batch_emb = encode_texts_mean_pooling(temp_model, batch_texts)

            if batch_emb.dim() > 3:
                batch_emb = batch_emb.squeeze()

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
            model, [query["query"]], clusters, 1
        )[0]
        first_cluster = clusters[cluster_ids[0]]
        result[query["qid"]] = first_cluster.get_topk_docids(model, query, docs, k)[0]
        print(f"{i}th query is done.")
    print("make_query_psuedo_answers finished.")
    return result
