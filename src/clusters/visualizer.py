# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import os
# from .cluster import Cluster
# from typing import List
# import json

# def visualize_clusters(
#     clusters: List[Cluster],
#     docs: dict,
#     save_path,
#     session_number,
#     representive_query_id=None,
#     representive_doc_ids=None,
# ):
#     all_points = []
#     all_labels = []
#     all_clusters = []

#     # 전체 임베딩 수집
#     for cluster_id, cluster in enumerate(clusters):
#         for doc_id in cluster.doc_ids:
#             emb = docs[doc_id]["TOKEN_EMBS"]
#             assert emb.shape == (254, 768), f"{doc_id} 임베딩 shape 오류: {emb.shape}"
#             pooled_embedding = emb.mean(dim=0)
#             all_points.append(pooled_embedding.cpu().numpy())
#             all_labels.append(doc_id)
#             all_clusters.append(cluster_id)

#     all_points = np.array(all_points)
#     all_clusters = np.array(all_clusters)
#     all_labels = np.array(all_labels)

#     # t-SNE로 차원 축소
#     tsne = TSNE(n_components=2, perplexity=30, random_state=42)
#     embeddings_2d = tsne.fit_transform(all_points)

#     # 색상 매핑
#     num_clusters = len(set(all_clusters))
#     cmap = plt.cm.get_cmap("tab10", num_clusters)
#     cluster_colors = [cmap(i) for i in all_clusters]

#     # 시각화
#     plt.figure(figsize=(14, 10))
#     plt.scatter(
#         embeddings_2d[:, 0],
#         embeddings_2d[:, 1],
#         c=cluster_colors,
#         s=10,
#         alpha=0.8,
#         edgecolors="black",
#         linewidths=0.3,
#     )

#     # 대표 문서만 라벨링
#     label_targets = set()
#     if representive_query_id:
#         label_targets.add(representive_query_id)
#     if representive_doc_ids:
#         label_targets.update(representive_doc_ids)

#     for i, doc_id in enumerate(all_labels):
#         if doc_id in label_targets:
#             if doc_id == representive_query_id:
#                 color = "red"
#             elif doc_id == representive_doc_ids[0]:
#                 color = "green"
#             else:
#                 color = "blue"
#             plt.text(
#                 embeddings_2d[i, 0],
#                 embeddings_2d[i, 1],
#                 doc_id,
#                 fontsize=7,
#                 fontweight="bold",
#                 color=color,
#                 alpha=0.8,
#             )

#     plt.title(
#         f"Session {session_number}"
#     )
#     plt.xlabel("t-SNE 1")
#     plt.ylabel("t-SNE 2")
#     plt.grid(True)
#     plt.tight_layout()

#     # 저장
#     plt.savefig(save_path, dpi=300)
#     print(f"[✓] 시각화 저장 완료: {os.path.abspath(save_path)}")

#     # 대표 query_id 및 doc_ids 저장
#     metadata = {
#         "session_number": session_number,
#         "query_id": representive_query_id,
#         "positive": representive_doc_ids[0] or [],
#         "negatives": representive_doc_ids[1:] or [],
#     }

#     meta_save_path = os.path.join(
#         os.path.dirname(save_path),
#         f"session_{session_number}_representatives.json"
#     )
#     with open(meta_save_path, "w", encoding="utf-8") as f:
#         json.dump(metadata, f, ensure_ascii=False, indent=4)

#     print(f"[✓] 대표 문서 정보 저장 완료: {os.path.abspath(meta_save_path)}")
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    # umap-learn 패키지가 설치되어 있어야 합니다: pip install umap-learn
    from umap import UMAP
    _UMAP_AVAILABLE = True
except Exception:
    _UMAP_AVAILABLE = False

from .cluster import Cluster


def visualize_clusters(
    clusters: List[Cluster],
    docs: dict,
    save_path: str,
    session_number: int,
    representive_query_id: Optional[str] = None,
    representive_doc_ids: Optional[list] = None,
    methods: Optional[list] = None,  # ["tsne", "pca", "umap"] 중 선택
):
    """
    clusters: Cluster 리스트 (각 Cluster는 .doc_ids 보유)
    docs: {doc_id: {"TOKEN_EMBS": torch.Tensor(shape=(254,768))}, ...}
    save_path: 저장할 기본 파일 경로(예: /path/plot.png) → _tsne/_pca/_umap 접미사로 저장됨
    session_number: 세션 번호
    representive_query_id: 라벨로 표시할 대표 쿼리 ID
    representive_doc_ids: [positive, negative1, negative2, ...] 형태
    methods: 생성할 차원축소 방법 리스트. 미지정 시 ["tsne", "pca", "umap"]
    """
    if methods is None:
        methods = ["tsne", "pca", "umap"]

    # 수집
    all_points = []
    all_labels = []
    all_clusters = []

    for cluster_id, cluster in enumerate(clusters):
        for doc_id in cluster.doc_ids:
            emb = docs[doc_id]["TOKEN_EMBS"]
            assert emb.shape == (254, 768), f"{doc_id} 임베딩 shape 오류: {emb.shape}"
            pooled_embedding = emb.mean(dim=0)
            all_points.append(pooled_embedding.detach().cpu().numpy())
            all_labels.append(doc_id)
            all_clusters.append(cluster_id)

    all_points = np.asarray(all_points)
    all_clusters = np.asarray(all_clusters)
    all_labels = np.asarray(all_labels)

    # 색상 매핑
    num_clusters = len(set(all_clusters))
    cmap = plt.cm.get_cmap("tab10", num_clusters)
    cluster_colors = [cmap(i) for i in all_clusters]

    # 라벨 대상 선정(안전 처리)
    label_targets = set()
    if representive_query_id:
        label_targets.add(representive_query_id)
    rep_docs = representive_doc_ids or []
    label_targets.update(rep_docs)
    positive_id = rep_docs[0] if len(rep_docs) > 0 else None
    negative_ids = set(rep_docs[1:]) if len(rep_docs) > 1 else set()

    # 파일 경로 분해
    base, ext = os.path.splitext(save_path)
    if not ext:
        ext = ".png"

    # 공통 그리기 함수
    def _plot_and_save(emb2d: np.ndarray, method_name: str):
        plt.figure(figsize=(14, 10))
        plt.scatter(
            emb2d[:, 0],
            emb2d[:, 1],
            c=cluster_colors,
            s=10,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.3,
        )

        # 대표 문서 라벨링
        for i, doc_id in enumerate(all_labels):
            if doc_id in label_targets:
                if doc_id == representive_query_id:
                    color = "red"
                elif positive_id and doc_id == positive_id:
                    color = "green"
                elif doc_id in negative_ids:
                    color = "blue"
                else:
                    color = "black"
                plt.text(
                    emb2d[i, 0],
                    emb2d[i, 1],
                    str(doc_id),
                    fontsize=7,
                    fontweight="bold",
                    color=color,
                    alpha=0.8,
                )

        plt.title(f"Session {session_number} – {method_name.upper()}")
        plt.xlabel(f"{method_name.upper()} 1")
        plt.ylabel(f"{method_name.upper()} 2")
        plt.grid(True)
        plt.tight_layout()

        out_path = f"{base}_{method_name}{ext}"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[✓] {method_name.upper()} 시각화 저장 완료: {os.path.abspath(out_path)}")

    # 각 방법으로 차원 축소
    # t-SNE perplexity는 샘플 수에 맞게 안전하게 보정
    n_samples = max(1, all_points.shape[0])
    safe_perplexity = min(30, max(5, min(n_samples - 1, 50)))

    for m in methods:
        m_lower = m.lower()
        if m_lower == "tsne":
            emb2d = TSNE(
                n_components=2,
                perplexity=safe_perplexity,
                random_state=42,
                init="pca",
                learning_rate="auto",
            ).fit_transform(all_points)
            _plot_and_save(emb2d, "tsne")

        elif m_lower == "pca":
            emb2d = PCA(n_components=2, random_state=42).fit_transform(all_points)
            _plot_and_save(emb2d, "pca")

        elif m_lower == "umap":
            if not _UMAP_AVAILABLE:
                print("[!] UMAP이 설치되어 있지 않습니다. `pip install umap-learn` 후 다시 실행하세요.")
                continue
            emb2d = UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                random_state=42,
            ).fit_transform(all_points)
            _plot_and_save(emb2d, "umap")

        else:
            print(f"[!] 지원하지 않는 방법입니다: {m}")

    # 대표 query_id 및 doc_ids 메타 저장(기존 로직 보강)
    metadata = {
        "session_number": session_number,
        "query_id": representive_query_id,
        "positive": [positive_id] if positive_id else [],
        "negatives": list(negative_ids) if negative_ids else [],
    }

    meta_save_path = os.path.join(
        os.path.dirname(save_path),
        f"session_{session_number}_representatives.json",
    )
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"[✓] 대표 문서 정보 저장 완료: {os.path.abspath(meta_save_path)}")
