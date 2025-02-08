from functions.similarities import calculate_S_qd_regl_dict

import torch
import numpy as np
from collections import defaultdict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import time
import math

MAX_SCORE = 999


def compute_sse_for_partition(
    partition, centroids, cluster_instances, device, batch_size=512
):
    sse = 0
    for k in partition:
        print(
            f"{device} | compute_sse_for_partition {k}, #instance {len(cluster_instances[k])}"
        )
        centroid = centroids[k].to(device, dtype=torch.float16)
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
            distances = distances**2
            sse += torch.sum(distances).item()
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
            distances_batch = distances_batch**2
            batch_distances.append(distances_batch)

        # 각 데이터 포인트에 대해 최소 거리 계산
        for i in range(batch_embeddings_tensor.shape[0]):
            min_distance = min(
                [distances_batch[i] for distances_batch in batch_distances]
            )
            distances.append(min_distance.item())

    return distances


def initialize_centroids(X, k, devices):
    print("Initialize centroid")
    n_samples = len(X)
    random_indices = np.random.randint(0, n_samples, size=k)
    centroids = [X[i]["LSH_MAPS"] for i in random_indices]
    # centroids = []
    # first_centroid = X[np.random.randint(0, n_samples)]["LSH_MAPS"]
    # centroids.append(first_centroid)
    # batch_size = 512

    # partitions = np.array_split(X, len(devices))
    # for _ in range(1, k):
    #     with concurrent.futures.ThreadPoolExecutor(
    #         max_workers=len(devices)
    #     ) as executor:
    #         distances_list = list(
    #             executor.map(
    #                 compute_distances_for_partition,
    #                 [
    #                     (partition, centroids, devices[idx], batch_size)
    #                     for idx, partition in enumerate(partitions)
    #                 ],
    #             )
    #         )
    #     distances = [distance for result in distances_list for distance in result]

    #     distances_tensor = torch.tensor(distances, device=devices[0])
    #     probabilities = distances_tensor / distances_tensor.sum()
    #     next_centroid_index = torch.multinomial(probabilities, num_samples=1).item()
    #     centroids.append(X[next_centroid_index]["LSH_MAPS"])
    return centroids


def create_centroid(instances, instance_ids):
    merged_hash = defaultdict(lambda: torch.zeros(768))
    # 254 * #instance_ids
    for _id in instance_ids:
        for key, value in instances[_id]["LSH_MAPS"].items():
            merged_hash[key] += value
    return merged_hash


def get_closest_clusters(args):
    partition, centroids, device, batch_size = args

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
                batch_clusters.append(distances)  # (batch_size,)
            distances_tensor = torch.stack(batch_clusters, dim=1)  # (batch, k)
            closest_batch_clusters = torch.argmin(distances_tensor, dim=1)  # (batch,)
            closest_clusters.extend(closest_batch_clusters.tolist())

        return closest_clusters


def kmeans_pp(X, k, max_iters, devices):
    centroids = initialize_centroids(X, k, devices)

    for iter_num in range(max_iters):
        print(f"Starting iteration {iter_num + 1} | centroids: #{len(centroids)}")
        start_time = time.time()
        batch_size = 512
        clusters = defaultdict(list)

        partitions = np.array_split(X, len(devices))
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            results = list(
                executor.map(
                    get_closest_clusters,
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


def find_closest_cluster_id(query, centroids, device):
    query_tokens = query["TOKEN_EMBS"]
    distances = []
    for centroid in centroids:
        distance = (
            MAX_SCORE - calculate_S_qd_regl_dict(query_tokens, centroid, device)
        ) ** 2
        distances.append(distance)
    closest_cluster = torch.argmin(torch.tensor(distances)).item()
    return closest_cluster


def calculate_mean(S1, N):
    mean_S1 = S1 / N
    return mean_S1


def calculate_rms(S1, S2, N):
    mean_S1 = S1 / N
    mean_S2 = S2 / N
    rms = math.sqrt(mean_S2 - mean_S1**2)
    return rms


def evict_cluster_instances(
    a, model, old_centroids, old_cluster_instances, old_centroids_statics
):
    # 오래된 문서 버리고, 현재 인코더로 재임베딩...?
    print("evict_cluster_instances started.")
    new_centroids, new_cluster_instances, new_centroids_statics = (
        [],
        defaultdict(list),
        {},
    )
    for old_idx, centroid in enumerate(old_centroids):
        MEAN = calculate_mean(
            S1=old_centroids_statics[old_idx]["S1"],
            N=old_centroids_statics[old_idx]["N"],
        )
        BOUNDARY = MEAN
        print(f"cluster {old_idx} | BOUNDARY: {BOUNDARY} | MEAN: {MEAN}")
        temp_cluster_instances = []
        if len(old_cluster_instances[old_idx]) != 0:
            tmp_centroids_statics = {"S1": 0, "S2": 0, "N": 0, "TOKEN_EMBS": None}
            for x in old_cluster_instances[old_idx]:
                x_dist = MAX_SCORE - calculate_S_qd_regl_dict(x["TOKEN_EMBS"], centroid)
                if x_dist <= BOUNDARY:
                    tmp_centroids_statics["S1"] = tmp_centroids_statics["S1"] + x_dist
                    tmp_centroids_statics["S2"] = (
                        tmp_centroids_statics["S2"] + x_dist**2
                    )
                    tmp_centroids_statics["N"] = tmp_centroids_statics["N"] + 1
                    tmp_centroids_statics["TOKEN_EMBS"] = x[
                        "TOKEN_EMBS"
                    ]  # 1개만 남을까봐
                    temp_cluster_instances.append(x)
        # evict 결과 유효한 클러스터이면 통계값/프로토타입/해당 클러스터 인스턴스 갱신
        if len(temp_cluster_instances) != 0:
            new_centroid = create_centroid(
                old_cluster_instances, [x["ID"] for x in temp_cluster_instances]
            )
            new_centroids.append(new_centroid)
            new_idx = len(new_centroids) - 1
            new_cluster_instances[new_idx] = temp_cluster_instances
            new_centroids_statics[new_idx] = tmp_centroids_statics
            print(f"cluster {old_idx} is alive.")
        # evict 결과 빈 클러스터이면 통계값/프로토타입/해당 클러스터 인스턴스 제거
        else:
            del old_centroids_statics[old_idx]
            del old_cluster_instances[old_idx]
            print(f"cluster {old_idx} is removed.")

    del old_centroids
    print("evict_cluster_instances ended.")
    return new_centroids, new_cluster_instances, new_centroids_statics


def get_initial_max_boundary(centroids_statics, closest_cluster, centroids):
    centroid_token_embs = centroids_statics[closest_cluster]["TOKEN_EMBS"]
    distances = []
    for i, centroid in enumerate(centroids):
        if i == closest_cluster:
            continue
        else:
            distances.append(
                MAX_SCORE - calculate_S_qd_regl_dict(centroid_token_embs, centroid)
            )
    min_dist = torch.min(distances)
    return min_dist


def assign_instance_or_centroid(
    centroids, centroids_statics, cluster_instances, current_session_data, t
):
    # 새로운 데이터를 클러스터에 추가, 새로운 데이터에 어느 클러스터에 할당되어있는지 추가
    print("assign_instance_or_centroid started.")
    X = list(current_session_data.values())

    for i, x in enumerate(X):
        distances = [
            MAX_SCORE - calculate_S_qd_regl_dict(x["TOKEN_EMBS"], centroid)
            for centroid in centroids
        ]
        closest_cluster = torch.argmin(distances).item()

        # 데이터 포인트 1개일 때 통계정보가 없으므로 가장 가까운 다른 클러스터까지의 거리로 max_boundary 휴리스틱으로 사용
        # 그렇지 않다면 RMS이내인지 학인, 아니면 새로운 centroid으로 할당
        s1, s2, n = (
            centroids_statics[closest_cluster]["S1"],
            centroids_statics[closest_cluster]["S2"],
            centroids_statics[closest_cluster]["N"],
        )
        if (
            n == 1
            and distances[closest_cluster]
            <= get_initial_max_boundary(centroids_statics, closest_cluster, centroids)
        ) or distances[closest_cluster] <= calculate_mean(
            S1=s1, N=n
        ) + t * calculate_rms(
            S1=s1, S2=s2, N=n
        ):
            cluster_instances[closest_cluster].append(x)
            centroids_statics[closest_cluster] = {
                "S1": centroids_statics[closest_cluster]["S1"]
                + distances[closest_cluster],
                "S2": centroids_statics[closest_cluster]["S2"]
                + distances[closest_cluster] ** 2,
                "N": centroids_statics[closest_cluster]["N"] + 1,
            }
        else:
            centroids.append(x["LSH_MAPS"])
            closest_cluster = len(centroids) - 1
            cluster_instances[closest_cluster].append(x)
            centroids_statics[closest_cluster] = {
                "S1": 0,
                "S2": 0,
                "N": 1,
                "TOKEN_EMBS": x["TOKEN_EMBS"],
            }
    print("assign_instance_or_centroid ended.")
    return centroids, centroids_statics, cluster_instances


def update_cluster_and_current_session_data(cluster_instances, current_session_data):
    print("update_cluster_and_current_session_data started.")
    # 필요없는 정보 제거, 가용 정보(현재 클러스터의 문서들+현재 세션 문서들) 갱신
    for key in cluster_instances:
        print(f"{key} : {len(cluster_instances[key])}")
        for x in cluster_instances[key]:
            if "LSH_MAPS" in x:
                del x["LSH_MAPS"]
            if "TOKEN_EMBS" in x:
                del x["TOKEN_EMBS"]
            current_session_data[x["ID"]] = x

    torch.cuda.empty_cache()
    print("update_cluster_and_current_session_data ended.")
    return cluster_instances, current_session_data
