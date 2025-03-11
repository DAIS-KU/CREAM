import torch
import faiss
import numpy as np
from loader import read_jsonl
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1)
    return sum_embeddings / sum_mask


def encode_texts(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            model_output = model(**inputs)
        pooled_embeddings = mean_pooling(model_output, inputs["attention_mask"])
        embeddings.append(pooled_embeddings.cpu().numpy())
    return np.vstack(embeddings)


def faiss_search(query_vectors, document_vectors, doc_ids, top_k):
    dim = query_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(document_vectors)
    distances, indices = index.search(query_vectors, top_k)
    return [[doc_ids[idx] for idx in indices[q]] for q in range(len(query_vectors))]


def multi_stage_faiss_search(
    queries, documents, query_batch_size=256, top_k=5, partition=100
):
    num_docs = len(documents)
    document_texts, doc_ids = [d["text"] for d in documents], [
        d["doc_id"] for d in documents
    ]
    query_texts = [q["query"] for q in queries]
    chunk_size = num_docs // partition
    top_100_per_query = [[] for _ in range(len(queries))]
    query_batches = []

    for q_start in range(0, len(queries), query_batch_size):
        q_end = min(q_start + query_batch_size, len(queries))
        query_chunk = query_texts[q_start:q_end]
        query_batches.append(encode_texts(query_chunk))

    for i in range(partition):
        print(f"Document partition {i+1}/{partition}")
        start, end = i * chunk_size, (i + 1) * chunk_size
        doc_chunk = document_texts[start:end]
        doc_vectors = encode_texts(doc_chunk)
        doc_ids_chunk = doc_ids[start:end]
        for j in range(len(query_batches)):
            print(f"Query partition {j+1}/{len(query_batches)}")
            query_batch = query_batches[j]
            top_cand = faiss_search(query_batch, doc_vectors, doc_ids_chunk, top_k)
            for q_idx in range(len(query_batch)):
                top_100_per_query[q_idx].extend(top_cand[q_idx])

    final_top_k_ids = []
    for q_idx, candidate_ids in enumerate(top_100_per_query):
        unique_ids = list(set(candidate_ids))
        candidate_texts = [
            document_texts[doc_ids.index(doc_id)] for doc_id in unique_ids
        ]
        candidate_vectors = encode_texts(candidate_texts)
        q_batch_idx = q_idx // query_batch_size
        q_inbatch_idx = q_idx % query_batch_size
        top_k_result = faiss_search(
            [query_batches[q_batch_idx][q_inbatch_idx]],
            candidate_vectors,
            unique_ids,
            top_k,
        )
        top_k_ids = [result[0] for result in top_k_result]  # (doc_id, score)
        final_top_k_ids.append(top_k_ids)

    return final_top_k_ids


def add_cosine_topk_answer():
    query_path = f"/mnt/DAIS_NAS/huijeong/train_session0_queries.jsonl"
    doc_path = f"/mnt/DAIS_NAS/huijeong/train_session0_docs.jsonl"
    queries = read_jsonl(query_path, True)
    documents = read_jsonl(doc_path, False)
    query_count = len(queries)
    doc_count = len(documents)
    print(f"Query count:{query_count}, Document count:{doc_count}")

    final_top_k_ids = multi_stage_faiss_search(queries, documents)
    for i in len(queries):
        queries[i]["cos_ans_pids"] = final_top_k_ids[i]
    save_jsonl(queries, "/mnt/DAIS_NAS/huijeong/train_session0_queries_cos.jsonl")


if __name__ == "__main__":
    add_cosine_topk_answer()
