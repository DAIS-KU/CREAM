import os
import random
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from umap import UMAP
    _UMAP_AVAILABLE = True
except Exception:
    _UMAP_AVAILABLE = False

from clusters import Cluster
from functions import calculate_S_qd_regl_batch_batch_batch  
import json

def visualize_clusters_pairwise_distance(
    clusters: List[Cluster],
    docs: dict,
    save_path: str,
    session_number: int,
    samples_per_cluster: int = 10,
    methods: Optional[list] = None,  # ["tsne", "pca", "umap"]
    device: Optional[torch.device] = None,
    batch_size: int = 8,            # dist_matrix 계산용 batch 크기
    representive_query_id: Optional[str] = None,
    representive_doc_ids: Optional[list] = None,
):
    """
    클러스터당 N개 샘플 → TOKEN_EMBS 기반 pairwise 유사도 → 거리=256-유사도
    → PCA / UMAP / t-SNE 시각화 (거리 행렬 기반).

    representive_query_id / representive_doc_ids:
      - 시각화 상에서 강조 표시 (색상 다르게 텍스트 라벨)
      - 별도 JSON 메타 파일로 저장
    """
    if methods is None:
        methods = ["tsne", "pca", "umap"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 클러스터별로 doc_id 샘플링
    sampled_doc_ids = []
    sampled_cluster_ids = []

    for cluster_id, cluster in enumerate(clusters):
        doc_ids = list(cluster.doc_ids)
        if not doc_ids:
            continue

        if len(doc_ids) > samples_per_cluster:
            selected = random.sample(doc_ids, samples_per_cluster)
        else:
            selected = doc_ids

        for doc_id in selected:
            if doc_id not in docs:
                continue
            emb = docs[doc_id]["TOKEN_EMBS"]
            if not isinstance(emb, torch.Tensor):
                raise TypeError(
                    f"{doc_id} TOKEN_EMBS 는 torch.Tensor 여야 합니다. 현재 타입: {type(emb)}"
                )
            sampled_doc_ids.append(doc_id)
            sampled_cluster_ids.append(cluster_id)

    if len(sampled_doc_ids) == 0:
        raise ValueError("샘플링된 문서가 없습니다. clusters / docs 를 확인하세요.")

    # 2. TOKEN_EMBS 스택
    token_embs = [docs[doc_id]["TOKEN_EMBS"] for doc_id in sampled_doc_ids]
    seq_lens = {emb.shape[0] for emb in token_embs}
    hidden_sizes = {emb.shape[1] for emb in token_embs}
    if len(seq_lens) != 1 or len(hidden_sizes) != 1:
        raise ValueError(
            f"모든 TOKEN_EMBS 의 shape 이 동일해야 합니다. "
            f"seq_lens={seq_lens}, hidden_sizes={hidden_sizes}"
        )

    E_all = torch.stack(token_embs, dim=0)  # (n_docs, L, 768)
    n_docs = E_all.shape[0]

    # 3. 배치 기반 거리 행렬 계산
    S_qd_matrix = calculate_S_qd_regl_batch_batch_batch(
        E_q=E_all,
        E_d=E_all,
        device=device,
        batch_size=batch_size,
    )  # (n_docs, n_docs)

    # 거리로 변환 (예: 256 - 유사도)
    dist_matrix = 256 - S_qd_matrix  # torch.Tensor (CPU)
    dist_matrix = dist_matrix.numpy()  # sklearn에 넘기기 위해 numpy로

    # 4. 색상 매핑 (클러스터용)
    num_clusters = len(set(sampled_cluster_ids))
    cmap = plt.cm.get_cmap("tab10", num_clusters)
    point_colors = [cmap(c) for c in sampled_cluster_ids]

    # 대표 query/doc 라벨용 정보
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

    def _plot_and_save(emb2d: np.ndarray, method_name: str):
        plt.figure(figsize=(14, 10))
        plt.scatter(
            emb2d[:, 0],
            emb2d[:, 1],
            c=point_colors,
            s=20,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
        )

        # 전체 doc 라벨을 찍되, 대표들은 색상 강조
        for i, doc_id in enumerate(sampled_doc_ids):
            if doc_id in label_targets:
                # 대표 쿼리/문서 색상 구분
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
                    alpha=0.9,
                )
            # else:
            #     # 나머지는 연한 회색으로 작게
            #     plt.text(
            #         emb2d[i, 0],
            #         emb2d[i, 1],
            #         str(doc_id),
            #         fontsize=6,
            #         color="gray",
            #         alpha=0.5,
            #     )

        plt.title(f"Session {session_number} – Pairwise distance ({method_name.upper()})")
        plt.xlabel(f"{method_name.upper()} 1")
        plt.ylabel(f"{method_name.upper()} 2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = f"{base}_pairwise_{method_name}{ext}"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[✓] {method_name.upper()} pairwise 시각화 저장 완료: {os.path.abspath(out_path)}")

    # 5. 차원 축소 및 시각화
    n_samples = dist_matrix.shape[0]
    safe_perplexity = min(30, max(5, min(n_samples - 1, 50)))

    for m in methods:
        m_lower = m.lower()

        if m_lower == "tsne":
            if n_samples <= 1:
                print("[!] 샘플 수가 1개 이하라 t-SNE를 수행할 수 없습니다.")
                continue

            emb2d = TSNE(
                n_components=2,
                perplexity=safe_perplexity,
                random_state=42,
                metric="precomputed",  # 거리 행렬 사용
                init="random",
                learning_rate="auto",
            ).fit_transform(dist_matrix)
            _plot_and_save(emb2d, "tsne")

        elif m_lower == "pca":
            emb2d = PCA(n_components=2, random_state=42).fit_transform(dist_matrix)
            _plot_and_save(emb2d, "pca")

        elif m_lower == "umap":
            if not _UMAP_AVAILABLE:
                print("[!] UMAP이 설치되어 있지 않습니다. `pip install umap-learn` 후 다시 실행하세요.")
                continue

            reducer = UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric="precomputed",
                random_state=42,
            )
            emb2d = reducer.fit_transform(dist_matrix)
            _plot_and_save(emb2d, "umap")

        else:
            print(f"[!] 지원하지 않는 방법입니다: {m}")

    # 6. 대표 query/doc 메타정보 JSON 저장
    metadata = {
        "session_number": session_number,
        "query_id": representive_query_id,
        "positive": [positive_id] if positive_id else [],
        "negatives": list(negative_ids) if negative_ids else [],
        "sampled_doc_ids": sampled_doc_ids,
    }

    meta_save_path = os.path.join(
        os.path.dirname(save_path),
        f"session_{session_number}_pairwise_representatives.json",
    )
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"[✓] 대표 문서 정보 저장 완료: {os.path.abspath(meta_save_path)}")

# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import List, Optional

# import torch
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

# try:
#     from umap import UMAP
#     _UMAP_AVAILABLE = True
# except Exception:
#     _UMAP_AVAILABLE = False

# from .cluster import Cluster


# def visualize_clusters(
#     clusters: List[Cluster],
#     docs: dict,
#     save_path: str,
#     session_number: int,
#     representive_query_id: Optional[str] = None,
#     representive_doc_ids: Optional[list] = None,
#     methods: Optional[list] = None,  # ["tsne", "pca", "umap"] 중 선택
# ):
#     """
#     clusters: Cluster 리스트 (각 Cluster는 .doc_ids 보유)
#     docs: {doc_id: {"TOKEN_EMBS": torch.Tensor(shape=(254,768))}, ...}
#     save_path: 저장할 기본 파일 경로(예: /path/plot.png) → _tsne/_pca/_umap 접미사로 저장됨
#     session_number: 세션 번호
#     representive_query_id: 라벨로 표시할 대표 쿼리 ID
#     representive_doc_ids: [positive, negative1, negative2, ...] 형태
#     methods: 생성할 차원축소 방법 리스트. 미지정 시 ["tsne", "pca", "umap"]
#     """
#     if methods is None:
#         methods = ["tsne", "pca", "umap"]

#     # 수집
#     all_points = []
#     all_labels = []
#     all_clusters = []

#     for cluster_id, cluster in enumerate(clusters):
#         for doc_id in cluster.doc_ids:
#             emb = docs[doc_id]["TOKEN_EMBS"]
#             assert emb.shape == (254, 768), f"{doc_id} 임베딩 shape 오류: {emb.shape}"
#             pooled_embedding = emb.mean(dim=0)
#             all_points.append(pooled_embedding.detach().cpu().numpy())
#             all_labels.append(doc_id)
#             all_clusters.append(cluster_id)

#     all_points = np.asarray(all_points)
#     all_clusters = np.asarray(all_clusters)
#     all_labels = np.asarray(all_labels)

#     # 색상 매핑
#     num_clusters = len(set(all_clusters))
#     cmap = plt.cm.get_cmap("tab10", num_clusters)
#     cluster_colors = [cmap(i) for i in all_clusters]

#     # 라벨 대상 선정(안전 처리)
#     label_targets = set()
#     if representive_query_id:
#         label_targets.add(representive_query_id)
#     rep_docs = representive_doc_ids or []
#     label_targets.update(rep_docs)
#     positive_id = rep_docs[0] if len(rep_docs) > 0 else None
#     negative_ids = set(rep_docs[1:]) if len(rep_docs) > 1 else set()

#     # 파일 경로 분해
#     base, ext = os.path.splitext(save_path)
#     if not ext:
#         ext = ".png"

#     # 공통 그리기 함수
#     def _plot_and_save(emb2d: np.ndarray, method_name: str):
#         plt.figure(figsize=(14, 10))
#         plt.scatter(
#             emb2d[:, 0],
#             emb2d[:, 1],
#             c=cluster_colors,
#             s=10,
#             alpha=0.8,
#             edgecolors="black",
#             linewidths=0.3,
#         )

#         # 대표 문서 라벨링
#         for i, doc_id in enumerate(all_labels):
#             if doc_id in label_targets:
#                 if doc_id == representive_query_id:
#                     color = "red"
#                 elif positive_id and doc_id == positive_id:
#                     color = "green"
#                 elif doc_id in negative_ids:
#                     color = "blue"
#                 else:
#                     color = "black"
#                 plt.text(
#                     emb2d[i, 0],
#                     emb2d[i, 1],
#                     str(doc_id),
#                     fontsize=7,
#                     fontweight="bold",
#                     color=color,
#                     alpha=0.8,
#                 )

#         plt.title(f"Session {session_number} – {method_name.upper()}")
#         plt.xlabel(f"{method_name.upper()} 1")
#         plt.ylabel(f"{method_name.upper()} 2")
#         plt.grid(True)
#         plt.tight_layout()

#         out_path = f"{base}_{method_name}{ext}"
#         plt.savefig(out_path, dpi=300)
#         plt.close()
#         print(f"[✓] {method_name.upper()} 시각화 저장 완료: {os.path.abspath(out_path)}")

#     # 각 방법으로 차원 축소
#     # t-SNE perplexity는 샘플 수에 맞게 안전하게 보정
#     n_samples = max(1, all_points.shape[0])
#     safe_perplexity = min(30, max(5, min(n_samples - 1, 50)))

#     for m in methods:
#         m_lower = m.lower()
#         if m_lower == "tsne":
#             emb2d = TSNE(
#                 n_components=2,
#                 perplexity=safe_perplexity,
#                 random_state=42,
#                 init="pca",
#                 learning_rate="auto",
#             ).fit_transform(all_points)
#             _plot_and_save(emb2d, "tsne")

#         elif m_lower == "pca":
#             emb2d = PCA(n_components=2, random_state=42).fit_transform(all_points)
#             _plot_and_save(emb2d, "pca")

#         elif m_lower == "umap":
#             if not _UMAP_AVAILABLE:
#                 print("[!] UMAP이 설치되어 있지 않습니다. `pip install umap-learn` 후 다시 실행하세요.")
#                 continue
#             emb2d = UMAP(
#                 n_components=2,
#                 n_neighbors=15,
#                 min_dist=0.1,
#                 random_state=42,
#             ).fit_transform(all_points)
#             _plot_and_save(emb2d, "umap")

#         else:
#             print(f"[!] 지원하지 않는 방법입니다: {m}")

#     # 대표 query_id 및 doc_ids 메타 저장(기존 로직 보강)
#     metadata = {
#         "session_number": session_number,
#         "query_id": representive_query_id,
#         "positive": [positive_id] if positive_id else [],
#         "negatives": list(negative_ids) if negative_ids else [],
#     }

#     meta_save_path = os.path.join(
#         os.path.dirname(save_path),
#         f"session_{session_number}_representatives.json",
#     )
#     with open(meta_save_path, "w", encoding="utf-8") as f:
#         json.dump(metadata, f, ensure_ascii=False, indent=4)

#     print(f"[✓] 대표 문서 정보 저장 완료: {os.path.abspath(meta_save_path)}")
