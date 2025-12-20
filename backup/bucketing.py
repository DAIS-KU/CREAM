from functions import (
    calculate_S_qd_regl,
    read_jsonl_as_dict,
    read_jsonl,
    save_jsonl,
    renew_data_mean_pooling,
)
from clusters import RandomProjectionLSH, renew_data
import random
import torch
import torch.nn.functional as F
from transformers import BertModel
import numpy as np
from sklearn.metrics import roc_auc_score


num_gpus = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]


def sampling_negs(pos_ids, doc_ids):
    neg_ids = random.sample(
        [doc_id for doc_id in doc_ids if doc_id not in pos_ids], len(pos_ids)
    )
    return neg_ids


def generate_pooling_data(qnt=20):
    source_path = "/home/work/.default/huijeong/data/lotte/"
    dest_path = "/home/work/.default/huijeong/data/pooled"
    domains = ["technology", "writing", "science", "recreation", "lifestyle"]
    pooled_query_path = f"{dest_path}/pooled_queries.jsonl"
    pooled_docs_path = f"{dest_path}/pooled_docs.jsonl"
    pooled_queries = []
    pooled_docs = []
    for domain in domains:
        print(f"Processing domain: {domain}")
        query_path = f"{source_path}/{domain}/{domain}_queries.jsonl"
        docs_path = f"{source_path}/{domain}/{domain}_docs.jsonl"
        queries = read_jsonl(query_path, True)
        queries = random.sample(queries, qnt)
        docs = read_jsonl_as_dict(docs_path, "doc_id")
        answer_docs = []
        for query in queries:
            answer_doc_ids = query["answer_pids"]
            answer_docs.extend(docs[doc_id] for doc_id in answer_doc_ids)
        pooled_queries.extend(queries)
        pooled_docs.extend(answer_docs)

    pooled_doc_ids = [doc["doc_id"] for doc in pooled_docs]
    for query in pooled_queries:
        neg_ids = sampling_negs(query["answer_pids"], pooled_doc_ids)
        query["neg_pids"] = neg_ids
    save_jsonl(pooled_queries, pooled_query_path)
    save_jsonl(pooled_docs, pooled_docs_path)


def model_builder(model_path):
    model = BertModel.from_pretrained("/home/work/.default/huijeong/bert_local").to(
        devices[-1]
    )
    if model_path:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def get_cosine_scores(q, d):
    q_norm = F.normalize(q, p=2, dim=0)
    d_norm = F.normalize(d, p=2, dim=0)
    return torch.sum(q_norm * d_norm).item()


def get_mean_pooling_similarities():
    path = "/home/work/.default/huijeong/data/pooled"
    queries = read_jsonl_as_dict(f"{path}/pooled_queries.jsonl", "qid")
    docs = read_jsonl(f"{path}/pooled_docs.jsonl", False)
    new_q_data, new_d_data = renew_data_mean_pooling(
        model_builder, None, list(queries.values()), docs
    )
    result = {}
    for q_data in list(new_q_data.values()):
        # print(f"q_data : {q_data}")
        q_id = q_data["doc_id"]
        pos_scores, neg_scores = [], []
        pos_ids, neg_ids = queries[q_id]["answer_pids"], queries[q_id]["neg_pids"]
        for pid in pos_ids:
            score = get_cosine_scores(q_data["EMB"], new_d_data[pid]["EMB"])
            pos_scores.append(score)
        for nid in neg_ids:
            score = get_cosine_scores(q_data["EMB"], new_d_data[nid]["EMB"])
            neg_scores.append(score)
        result[q_id] = {"pos_scores": pos_scores, "neg_scores": neg_scores}
    return result


def get_hash_similarties(nbits):
    path = "/home/work/.default/huijeong/data/pooled"
    queries = read_jsonl_as_dict(f"{path}/pooled_queries.jsonl", "qid")
    docs = read_jsonl(f"{path}/pooled_docs.jsonl", False)

    random_vectors = torch.randn(nbits, 768)
    hash = RandomProjectionLSH(
        random_vectors=random_vectors, embedding_dim=768, use_tensor_key=True
    )

    new_q_data, new_d_data = renew_data(
        queries=list(queries.values()),
        documents=docs,
        model_path=None,
        nbits=nbits,
        renew_q=True,
        renew_d=True,
        use_tensor_key=True,
    )
    result = {}
    for q_data in list(new_q_data.values()):
        # print(f"q_data : {q_data}")
        q_id = q_data["doc_id"]
        pos_scores, neg_scores = [], []
        pos_ids, neg_ids = queries[q_id]["answer_pids"], queries[q_id]["neg_pids"]
        for pid in pos_ids:
            # hashed_emb = hash.encode(new_d_data[pid]["TOKEN_EMBS"])
            score = calculate_S_qd_regl(
                q_data["TOKEN_EMBS"], new_d_data[pid]["TOKEN_EMBS"], devices[-1]
            )
            pos_scores.append(score)
        for nid in neg_ids:
            # hashed_emb = hash.encode(new_d_data[nid]["TOKEN_EMBS"])
            score = calculate_S_qd_regl(
                q_data["TOKEN_EMBS"], new_d_data[nid]["TOKEN_EMBS"], devices[-1]
            )
            neg_scores.append(score)
        result[q_id] = {"pos_scores": pos_scores, "neg_scores": neg_scores}
    return result


def _as_np(x):
    """
    list / numpy / torch 텐서를 모두 안전하게 numpy(float, 1D)로 변환
    - GPU 텐서는 CPU로 옮긴 뒤 변환
    - list/tuple of tensors도 안전 처리
    """
    # 1) torch.Tensor
    if isinstance(x, torch.Tensor):
        # 0-dim 텐서도 ravel()로 1D로
        return (
            x.detach().to("cpu", dtype=torch.float32, non_blocking=True).numpy().ravel()
        )
    # 2) list/tuple
    if isinstance(x, (list, tuple)):
        if len(x) > 0 and all(isinstance(t, torch.Tensor) for t in x):
            # 텐서 리스트 → 단일 텐서로 결합 후 변환
            # (shape 안 맞으면 torch.stack 대신 torch.cat/flatten 등으로 조정)
            try:
                xt = torch.stack(
                    [
                        t.detach().to("cpu", dtype=torch.float32, non_blocking=True)
                        for t in x
                    ]
                )
            except RuntimeError:
                # shape이 안 맞아 stack 불가 → 개별 flatten 후 cat
                xt = torch.cat(
                    [
                        t.detach()
                        .to("cpu", dtype=torch.float32, non_blocking=True)
                        .reshape(-1)
                        for t in x
                    ]
                )
            return xt.numpy().ravel()
        else:
            return np.asarray(x, dtype=float).ravel()
    # 3) 이미 numpy이거나 숫자/리스트 등
    return np.asarray(x, dtype=float).ravel()


def _cohens_d(pos, neg):
    """Cohen's d (pooled SD, unbiased sample var 사용)"""
    pos = _as_np(pos)
    neg = _as_np(neg)
    n1, n2 = len(pos), len(neg)
    if n1 == 0 or n2 == 0:
        return np.nan
    m1, m2 = pos.mean(), neg.mean()
    # ddof=1 → 표본분산
    v1, v2 = pos.var(ddof=1) if n1 > 1 else 0.0, neg.var(ddof=1) if n2 > 1 else 0.0
    # pooled standard deviation
    denom = (n1 - 1) * v1 + (n2 - 1) * v2
    if (n1 + n2 - 2) > 0:
        sp = np.sqrt(denom / (n1 + n2 - 2))
    else:
        sp = 0.0
    if sp == 0:
        # 분산이 0이면 평균이 같을 때 0, 다르면 부호만 가진 큰 값 대신 np.inf로 표시
        return 0.0 if np.isclose(m1, m2) else (np.sign(m1 - m2) * np.inf)
    return (m1 - m2) / sp


def _auc(pos, neg):
    """ROC AUC: pos=1, neg=0 레이블로 계산"""
    pos = _as_np(pos)
    neg = _as_np(neg)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    y_score = np.concatenate([pos, neg])
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        # 점수가 전부 같거나 수치 문제 발생 시 0.5로 처리(무분리)
        return 0.5


def compute_auc_and_cohensd(data):
    """
    data: {qid: {"pos_scores": [...], "neg_scores": [...]}}

    return:
      {
        "per_qid": {qid: {"auc": float, "cohens_d": float}},
        "summary": {
            "auc_mean": float, "auc_std": float,
            "d_mean": float, "d_std": float,
            "n_qids": int
        }
      }
    """
    auc_list, d_list = [], []

    for qid, v in data.items():
        pos = v.get("pos_scores", [])
        neg = v.get("neg_scores", [])
        auc = _auc(pos, neg)
        d = _cohens_d(pos, neg)
        if not np.isnan(auc):
            auc_list.append(auc)
        if not np.isnan(d):
            d_list.append(d)

    summary = {
        "auc_mean": float(np.mean(auc_list)) if auc_list else np.nan,
        "auc_std": (
            float(np.std(auc_list, ddof=1))
            if len(auc_list) > 1
            else 0.0 if auc_list else np.nan
        ),
        "d_mean": float(np.mean(d_list)) if d_list else np.nan,
        "d_std": (
            float(np.std(d_list, ddof=1))
            if len(d_list) > 1
            else 0.0 if d_list else np.nan
        ),
    }
    return summary


def compare(generate_data=False):
    if generate_data:
        generate_pooling_data()
    # sim_data = get_mean_pooling_similarities()
    # summary = compute_auc_and_cohensd(sim_data)
    # print(f"summary: {summary}\n")

    sim_data = get_hash_similarties(nbits=0)
    summary = compute_auc_and_cohensd(sim_data)
    print(f"summary(0bit): {summary}\n")

    # sim_data = get_hash_similarties(nbits=3)
    # summary = compute_auc_and_cohensd(sim_data)
    # print(f"summary(3bit): {summary}\n")

    # sim_data = get_hash_similarties(nbits=6)
    # summary = compute_auc_and_cohensd(sim_data)
    # print(f"summary(6bit): {summary}\n")

    # sim_data = get_hash_similarties(nbits=9)
    # summary = compute_auc_and_cohensd(sim_data)
    # print(f"summary(9bit): {summary}\n")

    # sim_data = get_hash_similarties(nbits=12)
    # summary = compute_auc_and_cohensd(sim_data)
    # print(f"summary(12bit): {summary}\n")
