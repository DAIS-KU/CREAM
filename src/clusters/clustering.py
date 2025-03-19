import concurrent.futures
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from .prototype import RandomProjectionLSH

from functions.similarities import calculate_S_qd_regl_dict, calculate_S_qd_regl_batch

MAX_SCORE = 256.0

num_gpus = 2  # torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]


def compute_sse_for_partition(
    partition, centroids, cluster_instances, device, batch_size=512
):
    sse = 0
    for k in partition:
        print(
            f"{device} | compute_sse_for_partition {k}, #instance {len(cluster_instances[k])}"
        )
        centroid = centroids[k]
        num_batches = (len(cluster_instances[k]) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            print(
                f"{device} | compute_sse_for_partition {k}, ({batch_idx}/{num_batches})"
            )
            batch = cluster_instances[k][
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ]
            batch_embeddings = [x["TOKEN_EMBS"] for x in batch]
            batch_embeddings_tensor = torch.stack(batch_embeddings).to(
                device, dtype=torch.float16
            )
            distances = MAX_SCORE - calculate_S_qd_regl_dict(
                batch_embeddings_tensor, centroid, device
            )
            print(
                f"compute_sse_for_partition | batch_embeddings_tensor: {batch_embeddings_tensor.shape}, distances:{distances.shape}"
            )
            distances = distances**2
            sse += torch.sum(distances).item()
    return sse


def compute_sse(centroids, cluster_instances):
    keys = list(cluster_instances.keys())
    chunk_size = (len(keys) + num_gpus - 1) // num_gpus
    partitions = [keys[i : i + chunk_size] for i in range(0, len(keys), chunk_size)]
    sse = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [
            executor.submit(
                compute_sse_for_partition,
                partition,
                centroids,
                cluster_instances,
                devices[idx],
            )
            for idx, partition in enumerate(partitions)
        ]
        for future in concurrent.futures.as_completed(futures):
            sse += future.result()
    return sse


def compute_distances_for_partition(args):
    X_partition, centroids, device, batch_size = args
    print(
        f"{device} compute_distances_for_partition | #X {len(X_partition)}, #centroids {len(centroids)}"
    )

    distances = []
    num_batches = (len(X_partition) + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        batch = X_partition[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_embeddings = [x["TOKEN_EMBS"] for x in batch]
        batch_embeddings_tensor = torch.stack(batch_embeddings).to(
            device
        )  # (batch_size, 768)

        batch_distances = []
        for centroid in centroids:
            distances_batch = MAX_SCORE - calculate_S_qd_regl_dict(
                batch_embeddings_tensor, centroid, device
            )  # (batch_size,)
            print(
                f"compute_distances_for_partition | batch_embeddings_tensor:{batch_embeddings_tensor.shape}, distances_batch:{distances_batch.shape}"
            )
            distances_batch = distances_batch**2
            batch_distances.append(distances_batch)

        for i in range(batch_embeddings_tensor.shape[0]):
            min_distance = min(
                [distances_batch[i] for distances_batch in batch_distances]
            )
            distances.append(min_distance.item())

    return distances


def initialize_centroids(X, k, lsh: RandomProjectionLSH):
    print("Initialize centroid")
    n_samples = len(X)
    random_indices = np.random.randint(0, n_samples, size=k)
    centroids = [lsh.encode(X[i]["TOKEN_EMBS"]) for i in random_indices]
    return centroids


def create_centroid(instances, instance_ids):
    merged_hash = defaultdict(lambda: torch.zeros(768).cpu())
    # 254 * #instance_ids
    for _id in instance_ids:
        for key, value in instances[_id]["LSH_MAPS"].items():
            merged_hash[key] += value.cpu()
    return merged_hash


def get_closest_clusters(partition, centroids, device, batch_size=128):
    with torch.cuda.device(device):
        closest_clusters = []
        num_batches = (len(partition) + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            print(f"{device} | get_closest_clusters batch ({batch_idx}/{num_batches})")
            batch = partition[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_tensor = torch.stack([x["TOKEN_EMBS"] for x in batch]).to(device)
            batch_clusters = []
            for centroid in centroids:
                distances = (
                    MAX_SCORE - calculate_S_qd_regl_dict(batch_tensor, centroid, device)
                ) ** 2
                # print(f"get_closest_clusters | batch_tensor: {batch_tensor.shape}, distances:{distances.shape}")
                batch_clusters.append(distances)  # (batch_size,)
            distances_tensor = torch.stack(batch_clusters, dim=1)  # (batch, k)
            closest_batch_clusters = torch.argmin(distances_tensor, dim=1)  # (batch,)
            closest_clusters.extend(closest_batch_clusters.tolist())
        return closest_clusters


def kmeans_pp(X, k, max_iters):
    centroids = initialize_centroids(X, k)

    for iter_num in range(max_iters):
        print(f"Starting iteration {iter_num + 1} | centroids: #{len(centroids)}")
        start_time = time.time()
        batch_size = 256
        centroid_batch_size = 32
        clusters = defaultdict(list)

        partitions = np.array_split(X, len(devices))
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            results = list(
                executor.map(
                    get_closest_clusters,
                    [
                        (
                            partition,
                            centroids,
                            devices[idx],
                            batch_size,
                            centroid_batch_size,
                        )
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
            new_centroid = create_centroid(X, cluster_indices)
            centroids.append(new_centroid)
        end_time = time.time()
        print(f"iter_num {iter_num} | execution_time: {end_time-start_time} sec.")

    cluster_instances = defaultdict(list)
    for k in sorted(clusters.keys()):
        cluster_instances[k] = [X[idx] for idx in clusters[k]]
        print(f"cluster {k} size: {len(cluster_instances[k])}")

    print(f"centroids : {len(centroids)}")
    return centroids, cluster_instances
