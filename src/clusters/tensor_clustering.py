import concurrent.futures
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from .prototype import RandomProjectionLSH
from functions.similarities import (
    calculate_S_qd_regl_dict,
    calculate_S_qd_regl_batch_batch,
    calculate_S_qd_regl_batch,
)

MAX_SCORE = 256.0

num_devices = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_devices)]


def initialize_centroids_use_tensor_key(X, k, lsh: RandomProjectionLSH):
    print("Initialize centroid")
    n_samples = len(X)
    random_indices = np.random.randint(0, n_samples, size=k)
    centroids = [lsh.encode(X[i]["TOKEN_EMBS"]) for i in random_indices]
    # centroids = [X[i]["LSH_MAPS"] for i in random_indices]
    return centroids


# def get_closest_clusters_use_tensor_key(args):
#     partition, centroids, device = args
#     batch_size = 512
#     with torch.cuda.device(device):
#         closest_clusters = []
#         num_batches = (len(partition) + batch_size - 1) // batch_size

#         for batch_idx in range(num_batches):
#             print(f"{device} | get_closest_clusters batch ({batch_idx}/{num_batches})")
#             batch = partition[batch_idx * batch_size : (batch_idx + 1) * batch_size]
#             batch_tensor = torch.stack([x["TOKEN_EMBS"] for x in batch]).to(device)
#             batch_clusters = []
#             for cidx, centroid in enumerate(centroids):
#                 print(f"ㄴ {device} | ({batch_idx}/{num_batches}) | {cidx}th centroid")
#                 distances = (
#                     MAX_SCORE
#                     - calculate_S_qd_regl_batch(
#                         batch_tensor, centroid.unsqueeze(0), device
#                     )
#                 ) ** 2
#                 if distances.dim() == 1:  # (batch_size, 1) 보장
#                     distances = distances.unsqueeze(1)
#                 batch_clusters.append(distances)
#             distances_tensor = torch.cat(
#                 batch_clusters, dim=1
#             )  # (batch_size, len(centroids))
#             closest_batch_clusters = torch.argmin(
#                 distances_tensor, dim=1
#             )  # (batch_size,)
#             closest_clusters.extend(closest_batch_clusters.tolist())
#         return closest_clusters


def create_centroid_use_tensor_key(
    instances, instance_ids, lsh: RandomProjectionLSH, nbits, batch_size=128
):
    chunk_cnt = min(num_devices, len(instance_ids))
    # stride, 순서보장 필요X
    batches = [instance_ids[i::chunk_cnt] for i in range(chunk_cnt)]

    # print(f"chunk_cnt: {chunk_cnt}, batches:{len(batches)}, len(instance_ids):{len(instance_ids)}")
    def process_on_device(device, split_ids):
        merged_tensor = torch.zeros(1 << nbits, 768).to(device)
        for i in range(0, len(split_ids), batch_size):
            batch_ids = split_ids[i : i + batch_size]
            batch_tensor = torch.stack(
                [lsh.encode(instances[_id]["TOKEN_EMBS"]) for _id in batch_ids], dim=0
            ).to(device)
            merged_tensor += torch.sum(batch_tensor, dim=0)
        return merged_tensor.cpu()

    results = []
    with ThreadPoolExecutor(max_workers=chunk_cnt) as executor:
        futures = [
            executor.submit(process_on_device, devices[i], batches[i])
            for i in range(chunk_cnt)
        ]
        results = [future.result() for future in futures]
    merged_tensor = sum(results).cpu()
    return merged_tensor


def get_closest_clusters_use_tensor_key(args):
    partition, centroids, device = args
    partition_bsz, centroid_bsz = 128, 8
    with torch.cuda.device(device):
        closest_clusters = []
        # 등분 갯수가 아니라 크기 중요, centroid_batches는 순서 보장도 필요
        partition_batches = [
            partition[i : i + partition_bsz]
            for i in range(0, len(partition), partition_bsz)
        ]
        centroid_batches = [
            centroids[i : i + centroid_bsz]
            for i in range(0, len(centroids), centroid_bsz)
        ]
        # (partition_batch,len(centroids))
        for batch_idx, partition_batch in enumerate(partition_batches):
            print(
                f"{device} | get_closest_clusters batch ({batch_idx}/{len(partition_batches)})"
            )
            batch_tensor = torch.stack([x["TOKEN_EMBS"] for x in partition_batch]).to(
                device
            )
            batch_clusters = []
            for centroid_batch in centroid_batches:
                batch_centroids_tensor = torch.stack(centroid_batch).to(device)
                distances = (
                    MAX_SCORE
                    - calculate_S_qd_regl_batch_batch(
                        batch_tensor, batch_centroids_tensor, device
                    )
                ) ** 2
                batch_clusters.append(distances)
            distances_tensor = torch.cat(
                batch_clusters, dim=1
            )  # (partition_bsz, len(centroids))
            closest_batch_clusters = torch.argmin(
                distances_tensor, dim=1
            )  # (partition_bsz,)
            closest_clusters.extend(closest_batch_clusters.tolist())
        print(f"closest_clusters: {len(closest_clusters)}")
        return closest_clusters


def kmeans_pp_use_tensor_key(X, k, max_iters, nbits, use_tensor_key=True):
    random_vectors = torch.randn(nbits, 768)
    lsh = RandomProjectionLSH(
        random_vectors=random_vectors,
        embedding_dim=768,
        use_tensor_key=use_tensor_key,
    )
    centroids = initialize_centroids_use_tensor_key(X, k, lsh)

    for iter_num in range(max_iters):
        print(f"Starting iteration {iter_num + 1} | centroids: #{len(centroids)}")
        start_time = time.time()
        clusters = defaultdict(list)
        # X순서보장 필요
        partitions = np.array_split(X, num_devices)
        with ThreadPoolExecutor(max_workers=num_devices) as executor:
            results = list(
                executor.map(
                    get_closest_clusters_use_tensor_key,
                    [
                        (partition, centroids, devices[idx])
                        for idx, partition in enumerate(partitions)
                    ],
                )
            )
        closest_clusters = [
            closest_cluster for result in results for closest_cluster in result
        ]
        unique_clusters = sorted(set(closest_clusters))
        cluster_to_idx = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
        for i, closest_cluster in enumerate(closest_clusters):
            clusters[cluster_to_idx[closest_cluster]].append(i)

        centroids = []
        for k in sorted(clusters.keys()):
            cluster_indices = clusters[k]
            print(f"clsuter {k} create new centroid.(instance {len(cluster_indices)})")
            new_centroid = create_centroid_use_tensor_key(
                X, cluster_indices, lsh, nbits
            )
            centroids.append(new_centroid)
        end_time = time.time()
        print(f"iter_num {iter_num} | execution_time: {end_time-start_time} sec.")

    cluster_instances = defaultdict(list)
    for k in sorted(clusters.keys()):
        cluster_instances[k] = [X[idx] for idx in clusters[k]]
        print(f"cluster {k} size: {len(cluster_instances[k])}")

    print(f"centroids : {len(centroids)}")
    return centroids, cluster_instances
