from functions.similarities import calculate_S_qd_regl_dict
from functions.utils import draw_elbow

import torch
import numpy as np
from collections import defaultdict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


def compute_sse_for_partition(partition, centroids, cluster_instances, device):
    sse = 0
    for k in partition:
        for x in cluster_instances[k]:
            sse += (
                1.0 - calculate_S_qd_regl_dict(x["TOKEN_EMBS"], centroids[k], device)
            ) ** 2
    return sse


def compute_sse(centroids, cluster_instances, devices):
    num_gpus = len(devices)
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
                devices[idx],
            )
            for idx, partition in enumerate(partitions)
        ]
        for future in concurrent.futures.as_completed(futures):
            sse += future.result()
    return sse


def compute_distances_for_partition(args):
    X_partition, centroids, device = args
    """특정 파티션에서 X와 centroids 간의 최소 거리 계산"""
    distances = []
    for x in X_partition:
        min_distance = float("inf")
        x_emb = x["TOKEN_EMBS"]
        for centroid in centroids:
            distance = (1.0 - calculate_S_qd_regl_dict(x_emb, centroid, device)) ** 2
            min_distance = min(min_distance, distance)
        distances.append(min_distance)
    return distances


def initialize_centroids(X, k, devices):
    print("Initialize centroid")
    n_samples = len(X)
    centroids = []
    first_centroid = X[np.random.randint(0, n_samples)]["LSH_MAPS"]
    centroids.append(first_centroid)

    partitions = np.array_split(X, len(devices))
    for _ in range(1, k):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(devices)
        ) as executor:
            distances_list = list(
                executor.map(
                    compute_distances_for_partition,
                    [
                        (partition, centroids, devices[idx])
                        for idx, partition in enumerate(partitions)
                    ],
                )
            )
        distances = [distance for result in distances_list for distance in result]

        distances_tensor = torch.tensor(distances, device=devices[0])
        probabilities = distances_tensor / distances_tensor.sum()
        next_centroid_index = torch.multinomial(probabilities, num_samples=1).item()
        centroids.append(X[next_centroid_index]["LSH_MAPS"])
    return centroids


def create_centroid(instances, instance_ids):
    instance_hashes = [instances[id]["LSH_MAPS"] for id in instance_ids]
    merged_hash = defaultdict(lambda: torch.zeros(768))
    for hash_map in instance_hashes:
        for key, value in hash_map.items():
            merged_hash[key] += value
    return merged_hash


def get_closest_clusters(args):
    partition, centroids, device = args
    with torch.cuda.device(device):
        closest_clusters = []
        for x in partition:
            distances = []
            for centroid in centroids:
                distance = (
                    1.0 - calculate_S_qd_regl_dict(x["TOKEN_EMBS"], centroid, device)
                ) ** 2
                distances.append(distance)
            closest_cluster = torch.argmin(torch.tensor(distances)).item()
            closest_clusters.append(closest_cluster)
        return closest_clusters


def kmeans_pp(X, k, max_iters, devices):
    centroids = initialize_centroids(X, k, devices)

    for iter_num in range(max_iters):
        print(f"Starting iteration {iter_num + 1}")
        clusters = defaultdict(list)

        partitions = np.array_split(X, len(devices))
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            results = list(
                executor.map(
                    get_closest_clusters,
                    [
                        (partition, centroids, devices[idx])
                        for idx, partition in enumerate(partitions)
                    ],
                )
            )
        closest_clusters = [
            closest_cluster for result in results for closest_cluster in result
        ]
        for i, closest_cluster in enumerate(closest_clusters):
            clusters[closest_cluster].append(i)

        new_centroids = []
        for k in sorted(clusters.keys()):
            cluster_indices = clusters[k]
            new_centroid = create_centroid(X, cluster_indices)
            new_centroids.append(new_centroid)
        centroids = new_centroids

    cluster_instances = defaultdict(list)
    for k in sorted(clusters.keys()):
        cluster_instances[k] = [X[idx] for idx in clusters[k]]

    return centroids, cluster_instances


def find_best_k(doc_data, start, end, gap, max_iters, devices):
    X = list(doc_data.values())
    print(len(X))
    sse = []
    k_values = range(start, end + 1, gap)
    for _k in k_values:
        centroids, cluster_instances = kmeans_pp(X, _k, max_iters, devices)
        _sse = compute_sse(centroids, cluster_instances, devices)
        sse.append(_sse)
        print(f"K:{_k} | _sse: {_sse}")
    draw_elbow(k_values, sse)


def find_closest_cluster_id(query, centroids, device):
    query_tokens = query["TOKEN_EMBS"]
    distances = []
    for centroid in centroids:
        distance = (1.0 - calculate_S_qd_regl_dict(query_tokens, centroid, device)) ** 2
        distances.append(distance)
    closest_cluster = torch.argmin(torch.tensor(distances)).item()
    return closest_cluster


def find_best_k_experiment(doc_path, start, end, gap, max_iters, devices):
    doc_data = read_jsonl(doc_path)

    doc_count = len(doc_data)
    print(f"Document count:{doc_count}")
    _, doc_data = renew_data(None, doc_data, 24, 768, False, True)
    print(f"doc_data: {len(doc_data)}")

    find_best_k(doc_data, start, end, gap, max_iters, devices)
