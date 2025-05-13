import json
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from transformers import BertModel, BertTokenizer

from data import read_jsonl, read_jsonl_as_dict, write_file
from functions import (
    InfoNCELoss,
    SimpleContrastiveLoss,
    evaluate_dataset,
    get_top_k_documents_by_cosine,
    renew_data_mean_pooling,
    get_top_k_documents,
)
from clusters import renew_data

torch.autograd.set_detect_anomaly(True)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


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


def train(dataset):
    domains = (
        [
            "technology",
            "writing",
            "lifestyle",
            "recreation",
            "science",
        ]
        if dataset == "lotte"
        else ["domain0", "domain1", "domain2", "domain3", "domain4"]
    )  #
    evaluate_term(dataset, domains, None, "None")
    for domain in domains:
        print(f"Train Dataset {dataset} | Domain {domain}")
        query_path = f"/home/work/retrieval/data/domain_dependency/{dataset}/train_{domain}_queries.jsonl"
        doc_path = f"/home/work/retrieval/data/domain_dependency/{dataset}/train_{domain}_docs.jsonl"
        queries = read_jsonl(query_path, True)
        docs = read_jsonl_as_dict(doc_path, "doc_id")
        doc_ids = list(docs.keys())

        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        model = BertModel.from_pretrained("bert-base-uncased").to(device)
        model.train()

        batch_size = 16
        loss_fn = InfoNCELoss()
        learning_rate = 2e-5
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_values = []

        total_loss, batch_cnt, query_cnt = 0, 1, len(queries)
        for start_idx in range(0, query_cnt, batch_size):
            end_idx = min(start_idx + batch_size, query_cnt)
            print(f"query {start_idx}-{end_idx}")

            query_batch, pos_docs_batch, neg_docs_batch = [], [], []
            for idx in range(start_idx, end_idx):
                query = queries[idx]
                pos_ids = random.sample(query["answer_pids"], 1)
                pos_docs = [docs[_id]["text"] for _id in pos_ids]
                neg_ids = random.sample(doc_ids, 6)
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
        method = f"base_model_{dataset}_{domain}"
        model_path = f"../data/domain_dependency/{method}.pth"
        torch.save(model.state_dict(), model_path)
        evaluate_term(dataset, domains, model_path, method)


def evaluate_term(dataset, domains, model_path, method):
    for domain in domains:
        print(f"Evaluate Dataset {dataset} | Domain {domain}")
        eval_query_path = (
            f"../data/domain_dependency/{dataset}/test_{domain}_queries.jsonl"
        )
        eval_doc_path = f"../data/domain_dependency/{dataset}/test_{domain}_docs.jsonl"

        eval_query_data = read_jsonl(eval_query_path, True)
        eval_doc_data = read_jsonl(eval_doc_path, False)

        eval_query_count = len(eval_query_data)
        eval_doc_count = len(eval_doc_data)
        print(
            f"{method}/{domain} | Query count:{eval_query_count}, Document count:{eval_doc_count}"
        )

        rankings_path = f"../data/rankings/{method}_{domain}.txt"

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

        rankings_path = f"../data/rankings/{method}_{domain}.txt"
        write_file(rankings_path, result)
        eval_log_path = f"../data/evals/{method}_{domain}.txt"
        evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
        del new_q_data, new_d_data
