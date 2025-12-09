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

import os
import random
import torch
from typing import List, Optional, Dict, Any


import glob
import json
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

try:
    from umap import UMAP

    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False


def dump_session_for_global_vis(
    session_number: int,
    clusters: List["Cluster"],
    docs: Dict[str, Dict[str, Any]],
    out_dir: str,
    samples_per_cluster: int = 10,
    representive_query_id: Optional[str] = None,
    representive_doc_ids: Optional[List[str]] = None,
):
    """
    - 각 세션별로:
      1) 클러스터에서 문서 샘플링
      2) TOKEN_EMBS를 스택해서 E_all (n_docs_session, L, H) 만들고
      3) session별 메타 + E_all을 .pt 로 저장
      4) 샘플링된 doc들 정보를 .jsonl로도 저장

    🔥 대표 query / positive / negative 는 무조건 샘플링에 포함되도록 보장
       (docs 안에 존재할 경우)

    representive_doc_ids:
      - [0] = positive, [1:] = negatives
    """
    os.makedirs(out_dir, exist_ok=True)

    sampled_doc_ids: List[str] = []
    sampled_cluster_ids: List[int] = []
    token_embs: List[torch.Tensor] = []

    # 1) 기본 클러스터별 랜덤 샘플링
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
            token_embs.append(emb)

    # 2) 대표 query / positive / negative 무조건 포함
    rep_docs = representive_doc_ids or []
    positive_id = rep_docs[0] if len(rep_docs) > 0 else None
    negative_ids = rep_docs[1:] if len(rep_docs) > 1 else []

    force_ids = set()
    if representive_query_id:
        force_ids.add(representive_query_id)
    force_ids.update(rep_docs)

    # 클러스터를 한번 훑어서 doc_id -> cluster_id 맵핑 만들어두면 편함
    doc_to_cluster: Dict[str, int] = {}
    for cluster_id, cluster in enumerate(clusters):
        for d in cluster.doc_ids:
            # 여러 클러스터에 있을 가능성은 낮다고 가정하고 처음 것만 사용
            if d not in doc_to_cluster:
                doc_to_cluster[d] = cluster_id

    for doc_id in force_ids:
        # docs에 없으면 스킵
        if doc_id not in docs:
            print(f"[!] Session {session_number}: 대표 doc_id {doc_id} 가 docs 에 없어 포함 불가")
            continue
        # 이미 샘플링 되어 있으면 패스
        if doc_id in sampled_doc_ids:
            continue

        emb = docs[doc_id]["TOKEN_EMBS"]
        if not isinstance(emb, torch.Tensor):
            raise TypeError(
                f"{doc_id} TOKEN_EMBS 는 torch.Tensor 여야 합니다. 현재 타입: {type(emb)}"
            )

        # 클러스터 정보 찾기 (없으면 -1 등으로 표시)
        cluster_id = doc_to_cluster.get(doc_id, -1)

        sampled_doc_ids.append(doc_id)
        sampled_cluster_ids.append(cluster_id)
        token_embs.append(emb)

    if len(token_embs) == 0:
        print(f"[!] Session {session_number}: 샘플링된 문서 없음, 건너뜀")
        return

    # 3) shape 체크
    seq_lens = {emb.shape[0] for emb in token_embs}
    hidden_sizes = {emb.shape[1] for emb in token_embs}
    if len(seq_lens) != 1 or len(hidden_sizes) != 1:
        raise ValueError(
            f"Session {session_number}: 모든 TOKEN_EMBS 의 shape 이 동일해야 합니다. "
            f"seq_lens={seq_lens}, hidden_sizes={hidden_sizes}"
        )

    E_all = torch.stack(token_embs, dim=0).cpu()  # CPU로 저장

    # 4) .pt 저장 (임베딩 + 메타)
    pt_path = os.path.join(out_dir, f"session_{session_number}_samples.pt")
    torch.save(
        {
            "session_number": session_number,
            "sampled_doc_ids": sampled_doc_ids,
            "sampled_cluster_ids": sampled_cluster_ids,
            "E_all": E_all,
            "representive_query_id": representive_query_id,
            "positive_id": positive_id,
            "negative_ids": negative_ids,
        },
        pt_path,
    )
    print(
        f"[✓] Session {session_number}: 샘플 {len(sampled_doc_ids)}개 .pt 저장 -> {os.path.abspath(pt_path)}"
    )

    # 5) .jsonl 저장 (어떤 샘플이 선택됐는지 기록)
    jsonl_path = os.path.join(out_dir, f"session_{session_number}_samples.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for doc_id, cluster_id in zip(sampled_doc_ids, sampled_cluster_ids):
            record = {
                "session_number": session_number,
                "doc_id": doc_id,
                "cluster_id": cluster_id,
                "is_query": bool(
                    representive_query_id and doc_id == representive_query_id
                ),
                "is_positive": bool(positive_id and doc_id == positive_id),
                "is_negative": bool(doc_id in negative_ids),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"[✓] Session {session_number}: 샘플 메타 .jsonl 저장 -> {os.path.abspath(jsonl_path)}"
    )


def global_pairwise_visualization_from_dumps(
    dump_dir: str,
    save_dir: str,
    methods: Optional[List[str]] = None,  # ["tsne", "pca", "umap"]
    device: Optional[torch.device] = None,
    batch_size: int = 8,
):
    """
    dump_dir 안의 session_*_samples.pt 들을:
      1) 로드해서 concat (전역 E_all)
      2) 전역 pairwise distance 계산
      3) TSNE/PCA/UMAP을 한 번씩만 실행
      4) 세션별 인덱스로 잘라서 plot
         - query_id: 빨강 (red)
         - positive_id: 초록 (green)
         - negatives: 파랑 (blue)
    """
    if methods is None:
        methods = ["tsne", "pca", "umap"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_dir, exist_ok=True)

    dump_paths = sorted(glob.glob(os.path.join(dump_dir, "session_*_samples.pt")))
    if not dump_paths:
        raise ValueError(f"{dump_dir} 에 session_*_samples.pt 파일이 없습니다.")

    all_E = []
    all_doc_ids: List[str] = []
    all_session_ids: List[int] = []
    all_cluster_keys: List[Tuple[int, int]] = []  # (session_number, cluster_id)
    session_to_indices: Dict[int, List[int]] = defaultdict(list)

    # 🔥 세션별 대표 정보
    session_query_id: Dict[int, Optional[str]] = {}
    session_positive_id: Dict[int, Optional[str]] = {}
    session_negative_ids: Dict[int, Set[str]] = {}

    for path in dump_paths:
        data = torch.load(path, map_location="cpu")
        session_number: int = data["session_number"]
        sampled_doc_ids: List[str] = data["sampled_doc_ids"]
        sampled_cluster_ids: List[int] = data["sampled_cluster_ids"]
        E_session: torch.Tensor = data["E_all"]

        rep_q = data.get("representive_query_id")
        pos_id = data.get("positive_id")
        neg_ids = data.get("negative_ids", [])

        session_query_id[session_number] = rep_q
        session_positive_id[session_number] = pos_id
        session_negative_ids[session_number] = set(neg_ids)

        n_session = E_session.shape[0]
        start_idx = len(all_doc_ids)

        all_E.append(E_session)
        all_doc_ids.extend(sampled_doc_ids)

        for local_idx in range(n_session):
            global_idx = start_idx + local_idx
            all_session_ids.append(session_number)
            all_cluster_keys.append((session_number, sampled_cluster_ids[local_idx]))
            session_to_indices[session_number].append(global_idx)

        print(
            f"[✓] 로드: session {session_number}, docs={n_session}, file={os.path.basename(path)}"
        )

    E_all = torch.cat(all_E, dim=0)  # (N_total, L, H)
    n_docs_total = E_all.shape[0]
    print(f"[✓] 전역 E_all shape: {E_all.shape}, 총 문서 수: {n_docs_total}")

    # 2) 전역 pairwise distance 계산
    S_qd_matrix = calculate_S_qd_regl_batch_batch_batch(
        E_q=E_all,
        E_d=E_all,
        device=device,
        batch_size=batch_size,
    )  # (N_total, N_total)
    print(f"[✓] 전역 pairwise distance 계산 완료")

    dist_matrix = 256 - S_qd_matrix
    dist_matrix = dist_matrix.cpu().numpy()

    # 3) 색상 (포인트 색) : cluster_id 기준
    #    - 같은 cluster_id 는 세션이 달라도 동일 색 사용
    all_cluster_ids = [cid for (_, cid) in all_cluster_keys]
    unique_cluster_ids = sorted(set(all_cluster_ids))
    cluster_id_to_idx = {cid: i for i, cid in enumerate(unique_cluster_ids)}
    num_clusters = len(unique_cluster_ids)

    cmap = plt.cm.get_cmap("tab20", num_clusters)
    point_colors = [cmap(cluster_id_to_idx[cid]) for cid in all_cluster_ids]

    n_samples = dist_matrix.shape[0]
    safe_perplexity = min(30, max(5, min(n_samples - 1, 50)))

    def _compute_emb2d(method: str) -> np.ndarray:
        print(f"Drawing in {method} started.")
        m_lower = method.lower()

        if m_lower == "tsne":
            if n_samples <= 1:
                raise ValueError("샘플 수가 1개 이하라 t-SNE를 수행할 수 없습니다.")
            return TSNE(
                n_components=2,
                perplexity=safe_perplexity,
                random_state=42,
                metric="precomputed",
                init="random",
                learning_rate="auto",
            ).fit_transform(dist_matrix)

        elif m_lower == "pca":
            return PCA(n_components=2, random_state=42).fit_transform(dist_matrix)

        elif m_lower == "umap":
            if not _UMAP_AVAILABLE:
                raise RuntimeError(
                    "UMAP이 설치되어 있지 않습니다. `pip install umap-learn` 후 다시 실행하세요."
                )
            reducer = UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric="precomputed",
                random_state=42,
            )
            return reducer.fit_transform(dist_matrix)

        else:
            raise ValueError(f"지원하지 않는 방법입니다: {method}")

    def _plot_single_session(
        emb2d_all: np.ndarray,
        method_name: str,
        axis_limits: Tuple[float, float, float, float],
        session_number: int,
    ):
        print(f"_plot_single_session in session {session_number} started.")
        indices = session_to_indices[session_number]
        if not indices:
            return

        emb2d = emb2d_all[indices]
        colors = [point_colors[i] for i in indices]
        doc_ids = [all_doc_ids[i] for i in indices]

        rep_q = session_query_id.get(session_number)
        pos_id = session_positive_id.get(session_number)
        neg_ids = session_negative_ids.get(session_number, set())

        label_targets = set()
        if rep_q:
            label_targets.add(rep_q)
        if pos_id:
            label_targets.add(pos_id)
        label_targets.update(neg_ids)

        plt.figure(figsize=(14, 10))
        plt.scatter(
            emb2d[:, 0],
            emb2d[:, 1],
            c=colors,  # 점 색: 클러스터/세션 기반
            s=20,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
        )

        # 🔥 라벨 색:
        #   - query_id -> red
        #   - positive_id -> green
        #   - negatives -> blue
        for i, doc_id in enumerate(doc_ids):
            if doc_id not in label_targets:
                continue

            if doc_id == rep_q:
                txt_color = "red"  # query
                fw = "bold"
            elif pos_id and doc_id == pos_id:
                txt_color = "green"  # positive
                fw = "bold"
            elif doc_id in neg_ids:
                txt_color = "blue"  # negatives
                fw = "bold"
            else:
                txt_color = "black"
                fw = "normal"

            plt.text(
                emb2d[i, 0],
                emb2d[i, 1],
                str(doc_id),
                fontsize=20,
                fontweight=fw,
                color=txt_color,
                alpha=0.9,
            )

        plt.title(f"Session {session_number} – Global pairwise ({method_name.upper()})")
        plt.xlabel(f"{method_name.upper()} 1")
        plt.ylabel(f"{method_name.upper()} 2")
        plt.grid(True, alpha=0.3)

        xmin, xmax, ymin, ymax = axis_limits
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        plt.tight_layout()

        out_path = os.path.join(
            save_dir,
            f"session_{session_number}_pairwise_{method_name}.png",
        )
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(
            f"[✓] Session {session_number} – {method_name.upper()} 시각화: {os.path.abspath(out_path)}"
        )

    # 4) method 별로 전역 축소 → 세션별 그림
    for m in methods:
        try:
            emb2d_all = _compute_emb2d(m)
        except Exception as e:
            print(f"[!] {m} 계산 중 오류: {e}")
            continue

        x_vals = emb2d_all[:, 0]
        y_vals = emb2d_all[:, 1]
        margin_x = (x_vals.max() - x_vals.min()) * 0.05
        margin_y = (y_vals.max() - y_vals.min()) * 0.05

        axis_limits = (
            x_vals.min() - margin_x,
            x_vals.max() + margin_x,
            y_vals.min() - margin_y,
            y_vals.max() + margin_y,
        )

        for session_number in session_to_indices.keys():
            _plot_single_session(
                emb2d_all=emb2d_all,
                method_name=m,
                axis_limits=axis_limits,
                session_number=session_number,
            )
