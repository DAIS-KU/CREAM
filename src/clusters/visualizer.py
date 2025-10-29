import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from .cluster import Cluster
from typing import List


def visualize_clusters(
    clusters: List[Cluster],
    docs: dict,
    save_path,
    session_number,
    representive_query_id=None,
    representive_doc_ids=None,
):
    all_points = []
    all_labels = []
    all_clusters = []

    # 전체 임베딩 수집
    for cluster_id, cluster in enumerate(clusters):
        for doc_id in cluster.doc_ids:
            emb = docs[doc_id]["TOKEN_EMBS"]
            assert emb.shape == (254, 768), f"{doc_id} 임베딩 shape 오류: {emb.shape}"
            pooled_embedding = emb.mean(dim=0)
            all_points.append(pooled_embedding.cpu().numpy())
            all_labels.append(doc_id)
            all_clusters.append(cluster_id)

    all_points = np.array(all_points)
    all_clusters = np.array(all_clusters)
    all_labels = np.array(all_labels)

    # t-SNE로 차원 축소
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(all_points)

    # 색상 매핑
    num_clusters = len(set(all_clusters))
    cmap = plt.cm.get_cmap("tab10", num_clusters)
    cluster_colors = [cmap(i) for i in all_clusters]

    # 시각화
    plt.figure(figsize=(14, 10))
    plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=cluster_colors,
        s=10,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3,
    )

    # 대표 문서만 라벨링
    label_targets = set()
    if representive_query_id:
        label_targets.add(representive_query_id)
    if representive_doc_ids:
        label_targets.update(representive_doc_ids)

    for i, doc_id in enumerate(all_labels):
        if doc_id in label_targets:
            if doc_id == representive_query_id:
                color = "red"
            elif doc_id == representive_doc_ids[0]:
                color = "green"
            else:
                color = "blue"
            plt.text(
                embeddings_2d[i, 0],
                embeddings_2d[i, 1],
                doc_id,
                fontsize=7,
                fontweight="bold",
                color=color,
                alpha=0.8,
            )

    plt.title(
        f"문장 임베딩 클러스터 시각화 (Mean Pooling 적용) Session {session_number}"
    )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.tight_layout()

    # 저장
    plt.savefig(save_path, dpi=300)
    print(f"[✓] 시각화 저장 완료: {os.path.abspath(save_path)}")
