from data import read_jsonl, read_jsonl_as_dict, preprocess, BM25Okapi
import matplotlib.pyplot as plt
import numpy as np
from clusters import renew_data
import torch
import torch.nn.functional as F
from functions import calculate_S_qd_regl_batch_batch


def draw_correltaion():
    pass


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


def get_correlation(q_batch_size=8, d_batch_size=64):
    eval_query_path = f"../data/datasetL/test_session0_queries.jsonl"
    eval_doc_path = f"../data/datasetL/test_session0_docs.jsonl"
    eval_queries = read_jsonl(eval_query_path, True)
    eval_docs = read_jsonl_as_dict(eval_doc_path, id_field="doc_id")
    filtered_doc_list = filter(eval_queries, eval_docs)

    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # 문서 임베딩 계산
    queries, docs = renew_data(queries=eval_queries, documents=filtered_doc_list)
    doc_vecs = torch.stack(
        [
            F.normalize(d["TOKEN_EMBS"].to(device).mean(dim=0), dim=0)
            for d in docs.values()
        ]
    )  # (N_docs, D)

    # 1. Mean pooling 코사인 유사도
    doc_vecs = torch.stack(
        [
            F.normalize(d["TOKEN_EMBS"].to(device).mean(dim=0), dim=0)
            for d in docs.values()
        ]
    )  # (D, 768)
    query_vecs = torch.stack(
        [
            F.normalize(q["TOKEN_EMBS"].to(device).mean(dim=0), dim=0)
            for q in queries.values()
        ]
    )  # (Q, 768)
    cosine_sims = torch.matmul(query_vecs, doc_vecs.T)  # (Q, D)
    sorted_cosine, _ = torch.sort(cosine_sims, descending=True, dim=1)
    avg_cosine_per_rank = sorted_cosine.mean(dim=0).cpu().tolist()

    # 2. S_qd 토큰기반 유사도 (배치처리)
    Q = len(eval_queries)
    D = len(filtered_doc_list)
    query_tokens = [q["TOKEN_EMBS"].to(device) for q in queries.values()]
    doc_tokens = [d["TOKEN_EMBS"].to(device) for d in docs.values()]
    S_qd_all = torch.zeros(Q, D)

    for i in range(0, Q, q_batch_size):
        q_batch = query_tokens[i : i + q_batch_size]  # list of (qlen, 768)
        q_tensor = torch.stack(q_batch)  # (q, qlen, 768)

        for j in range(0, D, d_batch_size):
            d_batch = doc_tokens[j : j + d_batch_size]  # list of (dlen, 768)
            d_tensor = torch.stack(d_batch)  # (d, dlen, 768)

            scores = calculate_S_qd_regl_batch_batch(
                q_tensor, d_tensor, device=device
            )  # (q, d)
            S_qd_all[i : i + q_tensor.size(0), j : j + d_tensor.size(0)] = scores.cpu()

    sorted_sqd = torch.sort(S_qd_all, descending=True, dim=1).values  # (Q, D)
    # normalized_sqd = sorted_sqd / (sorted_sqd.max(dim=1, keepdim=True).values + 1e-8)
    # avg_sqd_per_rank = normalized_sqd.mean(dim=0).tolist()
    avg_sqd_per_rank = sorted_sqd.mean(dim=0).tolist()

    plt.figure(figsize=(10, 5))
    plt.plot(avg_cosine_per_rank, label="Cosine (mean pooling)")
    plt.plot(avg_sqd_per_rank, label="S_qd (not normalized)", linestyle="--")
    plt.title("Average Similarity per Rank")
    plt.xlabel("Rank (0 = most similar)")
    plt.ylabel("Mean Similarity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rank_similarity_comparison2.png")
    print("✅ Saved to rank_similarity_comparison2.png")
