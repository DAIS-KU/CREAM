from data import read_jsonl, read_jsonl_as_dict, preprocess, BM25Okapi
import matplotlib.pyplot as plt
import numpy as np
from clusters import renew_data
import torch
import torch.nn.functional as F
from functions import calculate_S_qd_regl_batch_batch
from scipy.stats import pearsonr, spearmanr


def filter(query_list, docs_dict, sampling_size_per_query=50):
    print(f"Documents are passing through BM25.")
    doc_list = list(docs_dict.values())
    corpus = [doc["text"] for doc in doc_list]
    doc_ids = [doc["doc_id"] for doc in doc_list]
    bm25 = BM25Okapi(corpus=corpus, tokenizer=preprocess)
    candidate_ids = set()
    for i, query in enumerate(query_list):
        print(f"{i}th query")
        query_tokens = preprocess(query["query"])
        scores = bm25.get_scores(query_tokens)
        top_k_indices = np.argsort(scores)[::-1][:sampling_size_per_query]
        candidate_ids.update([doc_ids[i] for i in top_k_indices])
    doc_list = [docs_dict[doc_id] for doc_id in candidate_ids]
    return doc_list


def get_cosine_recall():
    eval_query_path = f"../data/datasetL/test_session0_queries.jsonl"
    eval_doc_path = f"../data/datasetL/test_session0_docs.jsonl"
    eval_queries = read_jsonl(eval_query_path, True)
    eval_docs = read_jsonl_as_dict(eval_doc_path, id_field="doc_id")
    filtered_doc_list = filter(eval_queries, eval_docs)  # list(eval_docs.values()) #

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    queries, docs = renew_data(queries=eval_queries, documents=filtered_doc_list)

    # 인덱스 정렬용
    all_doc_ids = [d["doc_id"] for d in filtered_doc_list]
    all_query_ids = list(queries.keys())
    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(all_doc_ids)}

    doc_vecs = torch.stack(
        [
            F.normalize(docs[doc_id]["TOKEN_EMBS"].to(device).mean(dim=0), dim=0)
            for doc_id in all_doc_ids
        ]
    )  # (D, 768)
    query_vecs = torch.stack(
        [
            F.normalize(queries[qid]["TOKEN_EMBS"].to(device).mean(dim=0), dim=0)
            for qid in all_query_ids
        ]
    )  # (Q, 768)
    cosine_sims = torch.matmul(query_vecs, doc_vecs.T)  # (Q, D)
    all_sims = cosine_sims.flatten().cpu().numpy()
    print(
        f"코사인 유사도 평균: {all_sims.mean():.4f}, 최소: {all_sims.min():.4f}, 최대: {all_sims.max():.4f}"
    )

    recall_at = [50, 100, 300, 500]
    recall_values = [0 for _ in recall_at]  # 전체 정답 개수
    valid_query_count = 0
    for q_idx, query in enumerate(eval_queries):
        answer_pids = set(query["answer_pids"])
        if not answer_pids:
            continue
        valid_query_count += 1
        top_indices = torch.argsort(cosine_sims[q_idx], descending=True)
        top_doc_ids = [all_doc_ids[j] for j in top_indices]
        for i, N in enumerate(recall_at):
            top_n_doc_ids = set(top_doc_ids[:N])
            hits = top_n_doc_ids.intersection(answer_pids)
            recall_values[i] += len(hits) / len(answer_pids)
    # for q_idx, query in enumerate(eval_queries[:10]):
    #     qid = all_query_ids[q_idx]
    #     answer_ids = query["answer_pids"]
    #     top_indices = torch.argsort(cosine_sims[q_idx], descending=True)
    #     top_doc_ids = [all_doc_ids[i] for i in top_indices[:10]]
    #     print(f"\n🔍 Query {qid}")
    #     print(f"정답 문서들: {answer_ids}")
    #     for aid in answer_ids:
    #         d_idx = doc_id_to_index.get(aid)
    #         if d_idx is not None:
    #             sim = cosine_sims[q_idx, d_idx].item()
    #             try:
    #                 rank = (top_indices == d_idx).nonzero(as_tuple=True)[0].item()
    #                 print(f" - 문서 {aid}: sim={sim:.4f}, rank={rank}")
    #             except:
    #                 print(f" - 문서 {aid}: sim={sim:.4f}, rank=100+")

    recalls = [recall_value / valid_query_count for recall_value in recall_values]
    for N, recall in zip(recall_at, recalls):
        print(f"Recall@{N}: {100 * recall:.3f}%")


def get_correlation_ans(q_batch_size=8, d_batch_size=64):
    eval_query_path = f"../data/datasetL/test_session0_queries.jsonl"
    eval_doc_path = f"../data/datasetL/test_session0_docs.jsonl"
    eval_queries = read_jsonl(eval_query_path, True)
    eval_docs = read_jsonl_as_dict(eval_doc_path, id_field="doc_id")

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    queries, docs = renew_data(queries=eval_queries, documents=list(eval_docs.values()))

    # 인덱스 정렬용
    all_doc_ids = list(docs.keys())
    all_query_ids = list(queries.keys())
    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(all_doc_ids)}

    # 임베딩 준비
    doc_vecs = torch.stack(
        [
            F.normalize(docs[doc_id]["TOKEN_EMBS"].to(device).mean(dim=0), dim=0)
            for doc_id in all_doc_ids
        ]
    )  # (D, 768)
    query_vecs = torch.stack(
        [
            F.normalize(queries[qid]["TOKEN_EMBS"].to(device).mean(dim=0), dim=0)
            for qid in all_query_ids
        ]
    )  # (Q, 768)

    # 1. 코사인 유사도
    cosine_sims = torch.matmul(query_vecs, doc_vecs.T)  # (Q, D)

    # 2. S_qd 점수 계산 (배치 처리)
    Q, D = len(all_query_ids), len(all_doc_ids)
    query_tokens = [queries[qid]["TOKEN_EMBS"].to(device) for qid in all_query_ids]
    doc_tokens = [docs[doc_id]["TOKEN_EMBS"].to(device) for doc_id in all_doc_ids]
    S_qd_all = torch.zeros(Q, D)

    for i in range(0, Q, q_batch_size):
        q_batch = query_tokens[i : i + q_batch_size]
        q_tensor = torch.stack(q_batch)

        for j in range(0, D, d_batch_size):
            d_batch = doc_tokens[j : j + d_batch_size]
            d_tensor = torch.stack(d_batch)
            scores = calculate_S_qd_regl_batch_batch(q_tensor, d_tensor, device=device)
            S_qd_all[i : i + q_tensor.size(0), j : j + d_tensor.size(0)] = scores.cpu()

    # 3. 정답 문서에 대한 유사도만 추출
    cosine_scores = []
    sqd_scores = []

    for q_idx, query in enumerate(eval_queries):
        answer_ids = query.get("answer_pids", [])
        for aid in answer_ids:
            d_idx = doc_id_to_index.get(aid, None)
            if d_idx is not None:
                cosine_scores.append(cosine_sims[q_idx, d_idx].item())
                sqd_scores.append(S_qd_all[q_idx, d_idx].item())

    # 4. 시각화
    cosine_scores = np.array(cosine_scores)
    sqd_scores = np.array(sqd_scores)
    pearson_corr, _ = pearsonr(cosine_scores, sqd_scores)
    spearman_corr, _ = spearmanr(cosine_scores, sqd_scores)

    plt.figure(figsize=(6, 6))
    plt.scatter(cosine_scores, sqd_scores, alpha=0.5)
    plt.title(
        f"Cosine vs S_qd\nPearson={pearson_corr:.4f}, Spearman={spearman_corr:.4f}"
    )
    plt.xlabel("Cosine Similarity (mean pooling)")
    plt.ylabel("S_qd Similarity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cosine_vs_sqd.png")
    print("✅ Saved to cosine_vs_sqd.png")
