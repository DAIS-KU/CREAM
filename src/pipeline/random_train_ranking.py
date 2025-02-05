import random
import torch
from transformers import BertModel, BertTokenizer

from functions import (
    InfoNCELoss,
    show_loss,
    get_top_k_documents,
    write_file,
    evaluate_dataset,
)
from data import read_jsonl, read_jsonl_as_dict, renew_data

import time

torch.autograd.set_detect_anomaly(True)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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


def session_train(query_path, model, doc_path, num_epochs):
    queries = read_jsonl(query_path)[:64]
    docs = read_jsonl_as_dict(doc_path, "doc_id")
    random.shuffle(queries)
    query_cnt = len(queries)
    loss_values = []
    validation_values = []
    accuracy_values = []

    current_device_index = torch.cuda.current_device()
    device = torch.device(f"cuda:{current_device_index}")

    loss_fn = InfoNCELoss()
    learning_rate = 2e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    batch_size = 32

    for epoch in range(num_epochs):
        total_loss, total_sec, batch_cnt = 0, 0, 0

        start_time = time.time()
        for start_idx in range(0, query_cnt, batch_size):
            end_idx = min(start_idx + batch_size, query_cnt)
            print(f"batch {start_idx}-{end_idx}")

            query_batch, pos_docs_batch, neg_docs_batch = [], [], []
            for qid in range(start_idx, end_idx):
                query = queries[qid]
                pos_docs = [
                    doc["text"] for doc in random.sample(list(docs.values()), 1)
                ]
                neg_docs = [
                    doc["text"] for doc in random.sample(list(docs.values()), 3)
                ]

                query_batch.append(query["query"])
                pos_embeddings = encode_texts(
                    model=model,
                    texts=pos_docs,
                )  # (positive_k, embedding_dim)
                pos_docs_batch.append(pos_embeddings)
                neg_embeddings = encode_texts(
                    model=model,
                    texts=neg_docs,
                )  # (negative_k, embedding_dim)
                neg_docs_batch.append(neg_embeddings)

            query_embeddings = encode_texts(
                model=model,
                texts=query_batch,
            )  # (batch_size, embedding_dim)
            positive_embeddings = torch.stack(
                pos_docs_batch
            )  # (batch_size, positive_k, embedding_dim)
            negative_embeddings = torch.stack(
                neg_docs_batch
            )  # (batch_size, negative_k, embedding_dim)
            print(
                f"query: {query_embeddings.shape}, pos: {positive_embeddings.shape} | negs: {negative_embeddings.shape}"
            )

            loss = loss_fn(query_embeddings, positive_embeddings, negative_embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss_values.append(loss.item())
            print(
                f"Processed {end_idx}/{query_cnt} queries | Batch Loss: {loss.item():.4f} | Total Loss: {total_loss / ((end_idx + 1) // batch_size):.4f}"
            )
            batch_cnt += 1
        end_time = time.time()
        execution_time = end_time - start_time
        total_sec += execution_time
        print(
            f"Epoch {epoch} | Total {total_sec} seconds, Avg {total_sec / batch_cnt} seconds."
        )

    return loss_values


def train(
    sesison_count=1,
    num_epochs=1,
    include_evaluate=True,
):
    total_loss_values = []
    for session_number in range(sesison_count):
        print(f"Training Session {session_number}")

        model = BertModel.from_pretrained("bert-base-uncased").to(devices[0])
        if session_number != 0:
            model_path = f"../data/model/gt_session_{session_number-1}.pth"
            model.load_state_dict(torch.load(model_path))
        model.train()
        new_model_path = f"../data/model/rand_session_{session_number}.pth"

        # 전처리가 필요한가...?
        loss_values = session_train(
            query_path=f"../data/sessions/train_session{session_number}_queries.jsonl",
            model=model,
            doc_path=f"../data/sessions/train_session{session_number}_docs.jsonl",
            num_epochs=num_epochs,
        )
        total_loss_values.extend(loss_values)
        show_loss(total_loss_values)
        torch.save(model.state_dict(), new_model_path)


def evaluate(sesison_count=1):
    for session_number in range(sesison_count):
        print(f"Evaluate Session {session_number}")
        eval_query_path = f"../data/sessions/test_session{session_number}_queries.jsonl"
        eval_doc_path = f"../data/sessions/test_session{session_number}_docs.jsonl"

        eval_query_data = read_jsonl(eval_query_path)[:10]
        eval_doc_data = read_jsonl(eval_doc_path)[:100]

        eval_query_count = len(eval_query_data)
        eval_doc_count = len(eval_doc_data)
        print(f"Query count:{eval_query_count}, Document count:{eval_doc_count}")

        rankings_path = f"../data/rankings/rand_{session_number}.txt"
        model_path = f"../data/model/rand_session_{session_number}.pth"

        start_time = time.time()
        new_q_data, new_d_data = renew_data(
            queries=eval_query_data,
            documents=eval_doc_data,
            nbits=0,
            embedding_dim=768,
            model_path=model_path,
            renew_q=True,
            renew_d=True,
        )
        end_time = time.time()
        print(f"Spend {end_time-start_time} seconds for encoding.")

        start_time = time.time()
        result = get_top_k_documents(new_q_data, new_d_data, k=10)
        end_time = time.time()
        print(f"Spend {end_time-start_time} seconds for retrieval.")

        rankings_path = f"../data/rankings/rand_{session_number}.txt"
        write_file(rankings_path, result)
        evaluate_dataset(eval_query_path, rankings_path, eval_doc_count)
        del new_q_data, new_d_data
