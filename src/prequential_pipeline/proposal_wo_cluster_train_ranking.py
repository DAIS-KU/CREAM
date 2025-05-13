import random
import time
from data import BM25Okapi, preprocess
import numpy as np

import torch
from transformers import BertModel, BertTokenizer

from ablation import get_samples_top_bottom
from clusters import renew_data
from data import load_eval_docs, read_jsonl, read_jsonl_as_dict, write_file
from functions import InfoNCELoss, evaluate_dataset, get_top_k_documents

torch.autograd.set_detect_anomaly(True)
tokenizer = BertTokenizer.from_pretrained("/home/work/retrieval/bert-base-uncased")

num_gpus = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]


def encode_texts(model, texts, max_length=256):
    device = model.device
    no_padding_inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    no_padding_inputs = {
        key: value.to(device) for key, value in no_padding_inputs.items()
    }
    outputs = model(**no_padding_inputs).last_hidden_state
    embedding = outputs[:, 0, :]  # [CLS]만 사용
    return embedding


def streaming_train(
    model,
    queries,
    doc_list,
    docs,
    ts,
    model_path,
    num_epochs,
    positive_k=1,
    negative_k=6,
    learning_rate=2e-5,
    batch_size=32,
):
    query_cnt = len(queries)
    loss_fn = InfoNCELoss()
    learning_rate = learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_values = []

    for epoch in range(num_epochs):
        total_loss, total_sec, batch_cnt = 0, 0, 0

        start_time = time.time()
        for start_idx in range(0, query_cnt, batch_size):
            end_idx = min(start_idx + batch_size, query_cnt)
            print(f"query {start_idx}-{end_idx}")

            query_batch, pos_docs_batch, neg_docs_batch = [], [], []
            for idx in range(start_idx, end_idx):
                query = queries[idx]
                pos_ids, neg_ids = get_samples_top_bottom(
                    query=query,
                    docs=docs,
                    clusters=clusters,
                    positive_k=positive_k,
                    negative_k=negative_k,
                    ts=ts,
                )
                pos_docs = [docs[_id]["text"] for _id in pos_ids]
                neg_docs = [docs[_id]["text"] for _id in neg_ids]

                query_batch.append(query["query"])
                query_embeddings = encode_texts(model=model, texts=query["query"])
                pos_embeddings = encode_texts(
                    model=model, texts=pos_docs
                )  # (positive_k, embedding_dim)
                pos_docs_batch.append(pos_embeddings)

                neg_embeddings = encode_texts(
                    model=model, texts=neg_docs
                )  # (negative_k, embedding_dim)
                neg_docs_batch.append(neg_embeddings)

            query_embeddings = encode_texts(
                model=model, texts=query_batch
            )  # (batch_size, embedding_dim)
            positive_embeddings = torch.stack(
                pos_docs_batch
            )  # (batch_size, positive_k, embedding_dim)
            negative_embeddings = torch.stack(
                neg_docs_batch
            )  # (batch_size, negative_k, embedding_dim)

            loss = loss_fn(query_embeddings, positive_embeddings, negative_embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss_values.append(loss.item())  # loss.item()
            batch_cnt += 1
            print(
                f"Processed {end_idx}/{query_cnt} queries | Batch Loss: {loss.item():.4f} | Total Loss: {total_loss / batch_cnt:.4f}"
            )
        end_time = time.time()
        execution_time = end_time - start_time
        total_sec += execution_time
        print(
            f"Epoch {epoch} | Total {total_sec} seconds, Avg {total_sec / batch_cnt} seconds."
        )
    return loss_values, ts


def filter_bm25(queries, docs, sampling_size_per_query=100):
    print(f"Documents are passing through BM25.")
    doc_list = list(docs.values())
    corpus = [doc["text"] for doc in doc_list]
    bm25 = BM25Okapi(corpus=corpus, tokenizer=preprocess)
    doc_ids = [doc["doc_id"] for doc in doc_list]
    candidate_ids = set()
    for i, query in enumerate(queries):
        print(f"{i}th query * {sampling_size_per_query}")
        query_tokens = preprocess(query["query"])
        scores = bm25.get_scores(query_tokens)
        top_k_indices = np.argsort(scores)[::-1][:sampling_size_per_query]
        candidate_ids.update([doc_ids[i] for i in top_k_indices])
    doc_list = [docs[doc_id] for doc_id in candidate_ids]
    return doc_list


def train(
    session_count=12,
    num_epochs=1,
    batch_size=32,
):
    for session_number in range(session_count):
        print(f"Train Session {session_number}")
        query_path = f"../data/sub/test_session{session_number}_queries.jsonl"
        doc_path = f"../data/sub/test_session{session_number}_docs.jsonl"
        queries = read_jsonl(query_path, True)
        docs = read_jsonl_as_dict(doc_path, "doc_id")
        ts = session_number

        model = BertModel.from_pretrained("/home/work/retrieval/bert-base-uncased").to(
            devices[1]
        )
        if session_number != 0:
            print("Load last session model.")
            model_path = (
                f"../data/model/proposal_wo_cluster_session_pre_{session_number-1}.pth"
            )
        else:
            print("Load Warming up model.")
            model_path = f"../data/base_model_lotte.pth"
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.train()
        new_model_path = (
            f"../data/model/proposal_wo_cluster_session_pre_{session_number}.pth"
        )
        doc_list = filter_bm25(queries, docs)
        loss_values = streaming_train(
            model, queries, doc_list, docs, ts, model_path, num_epochs
        )
        torch.save(model.state_dict(), new_model_path)


def evaluate(session_number, model_path):
    print(f"Evaluate Session {session_number}")
    eval_query_path = f"../data/sub/test_session{session_number}_queries.jsonl"
    eval_doc_path = f"../data/sub/test_session{session_number}_docs.jsonl"

    eval_query_data = read_jsonl(eval_query_path, True)
    eval_doc_data = read_jsonl(eval_doc_path, False)

    eval_query_count = len(eval_query_data)
    eval_doc_count = len(eval_doc_data)
    print(f"Query count:{eval_query_count}, Document count:{eval_doc_count}")

    start_time = time.time()
    new_q_data, new_d_data = renew_data(
        queries=eval_query_data,
        documents=eval_doc_data,
        model_path=model_path,
        nbits=12,
        renew_q=True,
        renew_d=True,
        use_tensor_key=True,
    )
    end_time = time.time()
    print(f"Spend {end_time-start_time} seconds for encoding.")

    start_time = time.time()
    result = get_top_k_documents(new_q_data, new_d_data)
    end_time = time.time()
    print(f"Spend {end_time-start_time} seconds for retrieval.")

    rankings_path = (
        f"../data/rankings/proposal_wo_cluster_session_pre_{session_number}.txt"
    )
    write_file(rankings_path, result)
    eval_log_path = (
        f"../data/evals/proposal_wo_cluster_session_pre_{session_number}.txt"
    )
    evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
    del new_q_data, new_d_data
