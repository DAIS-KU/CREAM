import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from .cluster import Cluster
from typing import List


def visualize_clusters(clusters: List[Cluster], docs: dict, save_path):
    all_points = []
    all_labels = []
    all_clusters = []
    for cluster_id, cluster in enumerate(clusters):
        for doc_id in cluster.doc_ids:
            emb = docs[doc_id]["TOKEN_EMBS"]
            assert emb.shape == (254, 768), f"{doc_id} 임베딩 shape 오류: {emb.shape}"
            pooled_embedding = emb.mean(dim=0)
            all_points.append(pooled_embedding.cpu().numpy())
            all_labels.append(doc_id)  # 문서 ID
            all_clusters.append(cluster_id)  # 클러스터 ID
    all_points = np.array(all_points)  # shape: (N, 768)
    all_clusters = np.array(all_clusters)  # shape: (N,)
    all_labels = np.array(all_labels)  # shape: (N,)
    # t-SNE로 차원 축소
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(all_points)  # shape: (N, 2)
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
    # 문서 ID 라벨 일부만 표시 (많을 경우 가독성 ↓)
    for i in range(0, len(all_labels), 50):  # 100개마다 하나씩 라벨링
        plt.text(
            embeddings_2d[i, 0],
            embeddings_2d[i, 1],
            all_labels[i],
            fontsize=6,
            alpha=0.6,
        )
    plt.title("문장 임베딩 클러스터 시각화 (Mean Pooling 적용)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.tight_layout()
    # 저장
    plt.savefig(save_path, dpi=300)
    print(f"[✓] 시각화 저장 완료: {os.path.abspath(save_path)}")
