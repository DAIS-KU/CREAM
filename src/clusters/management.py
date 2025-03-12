from typing import List

from data import Stream

from .cluster import Cluster
from .clustering import kmeans_pp
from .prototype import RandomProjectionLSH


def initialize(stream: Stream, k, nbits, max_iters=5) -> List[Cluster]:
    _, initial_docs = renew_data(
        queries=None,
        documents=stream.initial_docs,
        nbits=nbits,
        embedding_dim=768,
        renew_q=False,
        renew_d=True,
    )
    centroids, cluster_instances = kmeans_pp(list(initial_docs.values()), k, max_iters)
    clusters = []
    for centroid, instances in zip(centroids, cluster_instances):
        clusters.append(Cluster(centroid, instances, 0))
    return clusters


def find_k_closest_cluster(model, text, clusters: List[Cluster], k) -> List[int]:
    token_embs = get_passage_embeddings(model, text)
    device = model.device
    distances = []
    for cluster in clusters:
        distances.append(
            calculate_S_qd_regl_dict(token_embs, cluster.prototype, device)
        )
    sorted_distances, sorted_indices = torch.sort(
        torch.stack(distances).squeeze(), descending=False
    )
    return sorted_indices[:k].tolist()


def assign_instance_or_add_cluster(
    model, lsh: RandomProjectionLSH, clusters: List[Cluster], docs, ts
):
    print("assign_instance_or_add_cluster started.")
    for i, doc in enumerate(docs):
        closest_cluster_id = find_k_closest_cluster(model, doc["text"], clusters, 1)
        closest_clsuter = clusters[closest_cluster_id]
        doc_embs = get_passage_embeddings(model, doc["text"])

        # 데이터 포인트 1개일 때 통계정보 없으므로 무조건 할당
        # 2개 이상부터 RMS이내인지 학인, 아니면 새로운 centroid으로 할당
        s1, s2, n = closest_clsuter.get_statistics()
        doc_hash = lsh.encode(doc_embs)
        if (
            n == 1
            or closest_clsuter.get_distance(doc_embs) <= closest_clsuter.get_boundary()
        ):
            closest_clsuter.assign(doc["doc_id"], doc_embs, doc_hash, ts)
        else:
            clusters.append(Cluster(doc_hash, [doc], ts))
    print("assign_instance_or_add_cluster finished.")
    return clusters


def get_samples(model, query, clusters, positive_k, negative_k):
    cluster_ids = find_k_closest_cluster(
        model, query["query"], clusters, positive_k + negative_k
    )
    positive_id = sorted_indices[0]
    negative_ids = sorted_indices[1:]
    print(f"positive_id:{positive_id} | negative_ids:{negative_ids}")

    positive_samples = []
    pos_docs = clusters[positive_id].get_topk_docids(query, docs, 1)

    negative_samples = []
    for neg_id in negative_ids:
        neg_docs = clusters[neg_id].get_topk_docids(query, docs, 1)
        negative_samples.extend(neg_docs)

    print(
        f" query: {query['qid']} | positive: {positive_samples} | negative:{negative_samples}"
    )
    return positive_samples, negative_samples


def evict_clusters(model, lsh, docs, clusters: List[Cluster]) -> List[Cluster]:
    print("evict_cluster_instances started.")
    masks = []
    for cluster in enumerate(clusters):
        is_alive = cluster.evict(model, lsh, docs)
        masks.append(is_alive)
    clusters = [clusters[i] for i in range(len(clusters)) if masks[i]]
    print(f"evict_cluster_instances finished. (left #{len(clusters)})")
    return clusters


def retrieve_top_k_docs_from_cluster(model, queries, clusters, docs, k=10):
    result = {}
    for query in queries:
        closest_cluster_id = find_k_closest_cluster(model, query["query"], clusters, 1)
        closest_cluster = clusters[closest_cluster_id]
        top_k_doc_ids = closest_cluster.get_topk_docids(model, query, docs, k)
        {query["qid"]: top_k_doc_ids}
    return result
