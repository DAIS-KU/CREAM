import concurrent.futures
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

num_gpus = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]


def compute_sse_for_partition(
    partition, centroids, cluster_instances, device, batch_size=512
):
    sse = 0
    centroids_tensor = torch.stack([centroids[k] for k in partition]).to(
        device
    )  # (k, dim)
    for k in partition:
        print(
            f"{device} | compute_sse_for_partition {k}, #instance {len(cluster_instances[k])}"
        )
        num_batches = (len(cluster_instances[k]) + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            print(
                f"{device} | compute_sse_for_partition {k}, ({batch_idx}/{num_batches})"
            )
            batch = cluster_instances[k][
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ]
            batch_tensor = torch.stack([x["EMB"] for x in batch]).to(
                device, dtype=torch.float16
            )
            batch = torch.nn.functional.normalize(batch, p=2, dim=-1)
            centroids_tensor = torch.nn.functional.normalize(
                centroids_tensor, p=2, dim=-1
            )
            distances = (
                1
                - F.cosine_similarity(
                    batch_tensor.unsqueeze(1), centroids_tensor.unsqueeze(0), dim=2
                )
            ) ** 2
            sse += torch.sum(distances).item()
    return sse


def compute_sse(centroids, cluster_instances, num_gpus, devices):
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


def compute_distances_for_partition(X_partition, centroids, device, batch_size=512):
    print(
        f"{device} compute_distances_for_partition | #X {len(X_partition)}, #centroids {len(centroids)}"
    )
    distances = []
    centroids_tensor = torch.stack(centroids).to(device)  # (k, dim)
    num_batches = (len(X_partition) + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        batch = X_partition[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_tensor = torch.stack([x["EMB"] for x in batch]).to(
            device
        )  # (batch_size, 768)
        batch = torch.nn.functional.normalize(batch, p=2, dim=-1)
        batch_tensor = torch.nn.functional.normalize(batch_tensor, p=2, dim=-1)
        distances_tensor = (
            1
            - F.cosine_similarity(
                batch_tensor.unsqueeze(1), centroids_tensor.unsqueeze(0), dim=2
            )
        ) ** 2
        min_distances = torch.min(distances_tensor, dim=1)[0]
        distances.extend(min_distances.tolist())
    return distances


def initialize_centroids(X, k):
    print("Initialize centroid")
    n_samples = len(X)
    random_indices = np.random.randint(0, n_samples, size=k)
    centroids = [(X[i]["EMB"]) for i in random_indices]
    # print(f"initialize_centroids centroids:{len(centroids)}/{centroids[0].shape}")
    return centroids


def create_centroid(instances, instance_ids):
    mean_emb = torch.mean(
        torch.stack([instances[_id]["EMB"] for _id in instance_ids], dim=0), dim=0
    )
    # print(f"create_centroid mean_emb:{mean_emb.shape}")
    if torch.isnan(mean_emb).all():
        print("create_centroid mean_emb is zero vector")
    return mean_emb


def get_closest_clusters(partition, centroids, device, batch_size=128):
    with torch.cuda.device(device):
        closest_clusters = []
        # print(f"get_closest_clusters centroids:{len(centroids)}/{centroids[0].shape}")
        centroids_tensor = torch.stack(centroids, dim=0).to(device)  # (k, dim)
        num_batches = (len(partition) + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            print(f"{device} | get_closest_clusters batch ({batch_idx}/{num_batches})")
            batch = partition[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_tensor = torch.stack([x["EMB"] for x in batch]).to(
                device
            )  # (batch_size, dim)
            print(
                f"get_closest_clusters batch_tensor:{batch_tensor.shape}, centroids_tensor:{centroids_tensor.shape}"
            )

            batch_tensor = torch.nn.functional.normalize(batch_tensor, p=2, dim=-1)
            centroids_tensor = torch.nn.functional.normalize(
                centroids_tensor, p=2, dim=-1
            )
            distances_tensor = 1.0 - F.cosine_similarity(
                batch_tensor.unsqueeze(1), centroids_tensor.unsqueeze(0), dim=2
            )  # (batch, k)
            closest_batch_clusters = torch.argmin(distances_tensor, dim=1)  # (batch,)
            closest_clusters.extend(closest_batch_clusters.tolist())
        return closest_clusters


def kmeans_mean_pooling(X, k, max_iters):
    centroids = initialize_centroids(X, k)

    for iter_num in range(max_iters):
        print(f"Starting iteration {iter_num + 1} | centroids: #{len(centroids)}")
        start_time = time.time()
        batch_size = 256
        clusters = defaultdict(list)

        partitions = np.array_split(X, len(devices))
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            results = list(
                executor.map(
                    lambda args: get_closest_clusters(*args),
                    [
                        (partition, centroids, devices[idx], batch_size)
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
