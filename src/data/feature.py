import json

import faiss
import numpy as np
import torch
from .loader import read_jsonl, save_jsonl
from transformers import AutoModel, AutoTokenizer

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


def print_batch_sz(name, batches):
    print(f"#{name}: {len(batches)}")
    for i, sublist in enumerate(batches):
        print(f"{i}: {len(sublist)}")


def faiss_search(query_vectors, document_vectors, doc_ids, top_k):
    dim = query_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(document_vectors)
    distances, indices = index.search(query_vectors, top_k)
    res = [[doc_ids[idx] for idx in indices[q]] for q in range(len(query_vectors))]
    return res


# 배치 크기대로 나누고, 나머지는 새로운 배치로 둠.
def split_into_batches(data, batch_size):
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def multi_stage_faiss_search(
    queries, documents, query_batch_size=300, top_k=10, partition=100
):
    num_docs = len(documents)
    document_texts, doc_ids = [d["text"] for d in documents], [
        d["doc_id"] for d in documents
    ]
    query_texts = [q["query"] for q in queries]

    top_100_per_query = [[] for _ in range(len(queries))]
    query_batches = []
    for q_batch in split_into_batches(query_texts, query_batch_size):
        # print(f"q_batch: {type(q_batch)}, {len(q_batch)}, {q_batch[0]}")
        query_batch = encode_texts(q_batch)
        query_batches.append(query_batch)
    doc_chunks = np.array_split(document_texts, partition)
    doc_id_chunks = np.array_split(doc_ids, partition)

    for i in range(partition):
        print(f"Document partition {i+1}/{partition}")
        doc_chunk = doc_chunks[i].tolist()
        doc_ids_chunk = doc_id_chunks[i]
        # print(f"doc_chunk: {type(doc_chunk)}, {len(doc_chunk)}, {doc_chunk[0]}")
        doc_vectors = encode_texts(doc_chunk)

        qid = 0
        for j, query_batch in enumerate(query_batches):
            print(f"Query partition {j+1}/{len(query_batches)}")
            top_cand = faiss_search(query_batch, doc_vectors, doc_ids_chunk, top_k)
            for q_idx, candidates in enumerate(top_cand):
                top_100_per_query[qid].extend(candidates)
                print(f"top_100_per_query[{qid}]: {len(top_100_per_query[q_idx])}")
                qid += 1

    final_top_k_ids = []
    for q_idx, unique_ids in enumerate(top_100_per_query):
        candidate_texts = [
            document_texts[doc_ids.index(doc_id)] for doc_id in unique_ids
        ]
        print(
            f"top_k_result(#{q_idx}), unique_ids:{len(unique_ids)}, candidate_texts:{len(candidate_texts)}"
        )
        candidate_vectors = encode_texts(candidate_texts)

        q_batch_idx = q_idx // query_batch_size
        q_inbatch_idx = q_idx % query_batch_size
        query_tensor = torch.tensor([query_batches[q_batch_idx][q_inbatch_idx]])

        top_k_result = faiss_search(
            query_tensor,
            candidate_vectors,
            unique_ids,
            top_k,
        )
        top_k_ids = [result[0] for result in top_k_result]  # (doc_id, score)
        final_top_k_ids.append(top_k_ids)
    print_batch_sz("final_top_k_ids", final_top_k_ids)
    return final_top_k_ids


def cat(file_path="/mnt/DAIS_NAS/huijeong/train_session0_queries_cos.jsonl"):
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:  # 3줄만 읽고 종료
                break
            data = json.loads(line)  # JSON 문자열을 딕셔너리로 변환
            print(data)  # 출력


def add_cosine_topk_answer():
    query_path = f"/mnt/DAIS_NAS/huijeong/train_session0_queries.jsonl"
    doc_path = f"/mnt/DAIS_NAS/huijeong/train_session0_docs.jsonl"
    queries = read_jsonl(query_path, True)
    documents = read_jsonl(doc_path, False)
    query_count = len(queries)
    doc_count = len(documents)
    print(f"Query count:{query_count}, Document count:{doc_count}")

    final_top_k_ids = multi_stage_faiss_search(queries, documents)
    for i in range(len(queries)):
        queries[i]["cos_ans_pids"] = final_top_k_ids[i]
    save_jsonl(queries, "/mnt/DAIS_NAS/huijeong/train_session0_queries_cos.jsonl")


def get_top_k_documents_by_cosine(queries, documents, top_k):
    result = {}
    final_top_k_ids = multi_stage_faiss_search(
        queries=queries, documents=documents, top_k=top_k
    )
    for i in range(len(queries)):
        result[queries[i]["qid"]] = final_top_k_ids[i]
    return result


# if __name__ == "__main__":
#     # add_cosine_topk_answer()
#     cat()
