import random
import time
from typing import List

import torch
from transformers import BertModel, BertTokenizer

from clusters import (
    Cluster,
    RandomProjectionLSH,
    assign_instance_or_add_cluster,
    evict_clusters,
    get_samples,
    initialize,
    retrieve_top_k_docs_from_cluster,
)
from data import Stream, read_jsonl, read_jsonl_as_dict, write_file, write_line
from functions import InfoNCELoss, evaluate_dataset

ts = 0

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
    embedding = outputs[:, 0, :]  # [CLS]만 사용(Term score..?)
    return embedding


def streaming_train(
    stream: Stream,
    clusters: List[Cluster],
    docs,
    model,
    lsh: RandomProjectionLSH,
    num_epochs,
    positive_k=1,
    negative_k=3,
    learning_rate=2e-5,
    batch_size=32,
    use_label=False,
):
    stream_size = stream.get_stream_size()

    loss_fn = InfoNCELoss()
    learning_rate = learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_values = []

    for epoch in range(num_epochs):
        total_loss, total_sec, batch_cnt = 0, 0, 0

        start_time = time.time()
        for start_idx in range(0, stream_size, batch_size):
            end_idx = min(start_idx + batch_size, stream_size)
            print(f"stream {start_idx}-{end_idx}")

            query_batch, pos_docs_batch, neg_docs_batch = [], [], []

            for idx in range(start_idx, end_idx):
                query, stream_docs = stream.get_stream(idx)
                # Assign
                assign_instance_or_add_cluster(model, lsh, clusters, stream_docs, ts)
                # Sampling
                positive_ids, negative_ids = get_samples(
                    model, query, clusters, positive_k, negative_k
                )
                if use_label:
                    pos_docs = random.sample(query["answer_pids"], 1)[0]
                else:
                    pos_docs = [docs[_id]["text"] for _id in positive_ids]
                neg_docs = [docs[_id]["text"] for _id in negative_ids]

                query_batch.append(query["query"])
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
        ts += 1
    return loss_values


def train(
    sesison_count=1,
    num_epochs=2,
    batch_size=32,
    use_label=False,
    warmingup_rate=0.2,
    negative_k=3,
    k=100,
    nbits=16,
):
    total_loss_values = []
    loss_values_path = "../data/loss/total_loss_values_proposal.text"

    random_vectors = torch.randn(nbits, 768)
    lsh = RandomProjectionLSH(
        random_vectors=random_vectors, embedding_dim=embedding_dim
    )
    for session_number in range(sesison_count):
        print(f"Training Session {session_number}")
        stream = Stream(
            session_number,
            f"/mnt/DAIS_NAS/huijeong/train_session{session_number}_queries.jsonl",
            f"/mnt/DAIS_NAS/huijeong/train_session{session_number}_docs.jsonl",
            warmingup_rate,
        )
        print(f"Session {session_number} | Document count:{len(stream.docs.keys())}")

        # Initial
        if session_number == 0:
            start_time = time.time()
            clusters = initialize(stream, k, nbits)
            end_time = time.time()
            print(
                f"Spend {end_time-start_time} seconds for clustering({len(clusters)}, {len(stream.initial_docs)}) warming up."
            )

        model = BertModel.from_pretrained("bert-base-uncased").to(devices[-1])
        if session_number != 0:
            model_path = f"../data/model/proposal_session_{session_number-1}.pth"
            model.load_state_dict(torch.load(model_path))
        model.train()
        new_model_path = f"../data/model/proposal_session_{session_number}.pth"

        # Increment
        loss_values = streaming_train(stream, clusters, docs, model, lsh, num_epochs)
        write_line(
            loss_values_path, f"{session_number}, {', '.join(map(str, loss_values))}"
        )
        torch.save(model.state_dict(), new_model_path)

        clusters = evaluate_with_cluster(session_number, clusters, new_model_path)
        # Evict
        evict_clusters(model, lsh, docs, clusters)


def evaluate_with_cluster(session_number, clusters, model_path) -> List[Cluster]:
    queries = read_jsonl(
        f"/mnt/DAIS_NAS/huijeong/test_session{session_number}_queries.jsonl", True
    )
    docs = read_jsonl(
        f"/mnt/DAIS_NAS/huijeong/test_session{session_number}_docs.jsonl", False
    )
    print(
        f"Evaluate session {session_number} | #Query:{len(queries)}, #Document:{len(docs)}"
    )

    # Assign
    model = BertModel.from_pretrained("bert-base-uncased").to(devices[-1])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    lsh = RandomProjectionLSH(
        random_vectors=random_vectors, embedding_dim=embedding_dim
    )
    assign_instance_or_add_cluster(model, lsh, clusters, stream_docs, ts)

    # Retrieve
    start_time = time.time()
    result = retrieve_top_k_docs_from_cluster(model, queries, clusters, docs, 10)
    end_time = time.time()
    print(f"Spend {end_time-start_time} seconds for retrieval.")

    rankings_path = f"../data/rankings/proposal_{session_number}_with_cluster.txt"
    write_file(rankings_path, result)
    eval_log_path = f"../data/evals/proposal_{session_number}_with_cluster.txt"
    evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
    return clusters
