import random
import torch
from transformers import BertModel, BertTokenizer
from functions import InfoNCELoss, evaluate_dataset, get_top_k_documents, write_file

from data import read_jsonl, write_file, renew_data
import time
from buffer import (
    Buffer,
    DataArguments,
    TevatronTrainingArguments,
    DenseModel,
    ModelArguments,
)

torch.autograd.set_detect_anomaly(True)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

num_gpus = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]


def encode_texts(model, texts, device, max_length=256):
    no_padding_inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    no_padding_inputs = {
        key: value.to(device) for key, value in no_padding_inputs.items()
    }
    outputs = model(**no_padding_inputs).last_hidden_state
    embedding = outputs[:, 0, :]
    return embedding


def get_top_k_documents_by_cosine(query_emb, current_data_embs, k=10):
    scores = torch.nn.functional.cosine_similarity(
        query_emb.unsqueeze(0), current_data_embs, dim=1
    )
    top_k_scores, top_k_indices = torch.topk(scores, k)
    return top_k_indices.tolist()


# https://github.com/caiyinqiong/L-2R/blob/main/src/tevatron/trainer.py
def session_train(queries, docs, method, model, buffer: Buffer, num_epochs):
    random.shuffle(queries)
    query_cnt = len(queries)
    loss_values = []
    current_device_index = torch.cuda.current_device()
    device = torch.device(f"cuda:{current_device_index}")

    loss_fn = InfoNCELoss()
    learning_rate = 2e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    batch_size = 32

    for epoch in range(num_epochs):
        total_loss = 0

        for start_idx in range(0, query_cnt, batch_size):
            end_idx = min(start_idx + batch_size, query_cnt)
            print(f"batch {start_idx}-{end_idx}")

            query_batch, pos_docs_batch, neg_docs_batch = [], [], []

            for qid in range(start_idx, end_idx):
                query = queries[qid]
                pos_docs = [
                    doc["text"] for doc in random.sample(list(docs.values()), 1)
                ]  # [get_top_k_documents_by_cosine(...)]
                neg_docs = [
                    doc["text"] for doc in random.sample(list(docs.values()), 3)
                ]  # buffer.retrieve(...)
                query_batch.append(query["query"])
                pos_embeddings = encode_texts(
                    model=model,
                    texts=pos_docs,
                    device=device,
                    max_length=256,
                    is_cls=True,
                )  # (positive_k, embedding_dim)
                pos_docs_batch.append(pos_embeddings)
                neg_embeddings = encode_texts(
                    model=model,
                    texts=neg_docs,
                    device=device,
                    max_length=256,
                    is_cls=True,
                )  # (negative_k, embedding_dim)
                neg_docs_batch.append(neg_embeddings)

            query_embeddings = encode_texts(
                model=model,
                texts=query_batch,
                device=device,
                max_length=256,
                is_cls=True,
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
    return loss_values


def train(method, session_count=1, num_epochs=1):
    for session_number in range(session_count):
        print(f"Train Session {session_number}")
        query_path = f"../data/sessions/train_session{session_number}_queries.jsonl"
        doc_path = f"../data/sessions/train_session{session_number}_docs.jsonl"

        query_data = read_jsonl(query_path)
        doc_data = read_jsonl(doc_path)

        model_args = ModelArguments(model_name_or_path="bert-base-uncased")
        output_dir = "../data/model"
        training_args = TevatronTrainingArguments(output_dir=output_dir)
        model = DenseModel.build(
            model_args,
            training_args,
            cache_dir=model_args.cache_dir,
        )
        if session_number != 0:
            model_path = f"../data/model/{method}_session_{session_number-1}.pth"
            model.load_state_dict(torch.load(model_path))
        new_model_path = f"../data/model/{method}_session_{session_number}.pth"
        model.train()
        # https://github.com/caiyinqiong/L-2R/blob/main/src/tevatron/arguments.py#L48
        # Buffer 객체 선언 시 내부엣 arg path에서 buffer.pkl 읽어오거나 메모리 콜렉션 초기화
        buffer = Buffer(model, tokenizer, DataArguments(), TevatronTrainingArguments())

        loss_values = session_train(
            query_data, doc_data, num_epochs, method, buffer, session_number
        )
        torch.save(model.state_dict(), new_model_path)
        # buffer.save(...) # L2R의 경우 buffer.replace()후 save()를 호출


def evaluate(method, sesison_count=1):
    for session_number in range(sesison_count):
        print(f"Evaluate Session {session_number}")
        eval_query_path = f"../data/sessions/test_session{session_number}_queries.jsonl"
        eval_doc_path = f"../data/sessions/test_session{session_number}_docs.jsonl"

        eval_query_data = read_jsonl(eval_query_path)[:10]
        eval_doc_data = read_jsonl(eval_doc_path)[:100]

        eval_query_count = len(eval_query_data)
        eval_doc_count = len(eval_doc_data)
        print(f"Query count:{eval_query_count}, Document count:{eval_doc_count}")

        rankings_path = f"../data/rankings/{method}_session_{session_number}.txt"
        model_path = f"../data/model/{method}_session_{session_number}.pth"

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

        rankings_path = f"../data/rankings/{method}_session_{session_number}.txt"
        write_file(rankings_path, result)
        evaluate_dataset(eval_query_path, rankings_path, eval_doc_count)
        del new_q_data, new_d_data
