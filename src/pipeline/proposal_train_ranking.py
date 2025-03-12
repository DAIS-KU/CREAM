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
    get_samples_and_weights,
    initialize,
    retrieve_top_k_docs_from_cluster,
)
from data import Stream, read_jsonl, read_jsonl_as_dict, write_file, write_line
from functions import InfoNCELoss, evaluate_dataset

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
    ts,
    stream: Stream,
    clusters: List[Cluster],
    model,
    lsh: RandomProjectionLSH,
    num_epochs,
    positive_k=1,
    negative_k=6,
    learning_rate=2e-5,
    batch_size=32,
    use_label=False,
    use_weight=False,
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
                print(f"{idx}th stream(#{len(stream_docs)})")
                # Assign
                assign_instance_or_add_cluster(
                    model, lsh, clusters, stream_docs, stream.docs, ts
                )
                # Sampling
                pos_ids, pos_weights, neg_ids, neg_weights = get_samples_and_weights(
                    model, query, stream.docs, clusters, positive_k, negative_k, ts
                )
                if use_label:
                    pos_docs = random.sample(query["answer_pids"], 1)[0]
                else:
                    pos_docs = [stream.docs[_id]["text"] for _id in pos_ids]
                neg_docs = [stream.docs[_id]["text"] for _id in neg_ids]

                query_batch.append(query["query"])
                pos_embeddings = encode_texts(
                    model=model, texts=pos_docs
                )  # (positive_k, embedding_dim)
                if use_weight:
                    pos_weights = torch.tensor(pos_weights).unsqueeze(
                        1
                    )  # (positive_k, 1)
                    pos_embeddings = pos_embeddings * pos_weights
                pos_docs_batch.append(pos_embeddings)

                neg_embeddings = encode_texts(
                    model=model, texts=neg_docs
                )  # (negative_k, embedding_dim)
                if use_weight:
                    neg_weights = torch.tensor(neg_weights).unsqueeze(
                        1
                    )  # (negative_k, 1)
                    neg_embeddings = neg_embeddings * neg_weights
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
                f"Processed {end_idx}/{stream_size} queries | Batch Loss: {loss.item():.4f} | Total Loss: {total_loss / batch_cnt:.4f}"
            )
        end_time = time.time()
        execution_time = end_time - start_time
        total_sec += execution_time
        print(
            f"Epoch {epoch} | Total {total_sec} seconds, Avg {total_sec / batch_cnt} seconds."
        )
    return loss_values, ts


def train(
    sesison_count=4,
    num_epochs=1,
    batch_size=32,
    warmingup_rate=0.2,
    negative_k=6,
    k=103,
    nbits=6,
    use_label=False,
    use_weight=False,
):
    ts = 0
    total_loss_values = []
    loss_values_path = "../data/loss/total_loss_values_proposal.text"

    random_vectors = torch.randn(nbits, 768)
    lsh = RandomProjectionLSH(random_vectors=random_vectors, embedding_dim=768)
    prev_docs = None
    for session_number in range(sesison_count):
        print(f"Training Session {session_number}")
        stream = Stream(
            session_number,
            f"/mnt/DAIS_NAS/huijeong/train_session{session_number}_queries.jsonl",
            f"/mnt/DAIS_NAS/huijeong/train_session{session_number}_docs.jsonl",
            warmingup_rate,
            prev_docs,
        )
        print(f"Session {session_number} | Document count:{len(stream.docs.keys())}")

        model = BertModel.from_pretrained("bert-base-uncased").to(devices[-1])
        if session_number != 0:
            model_path = f"../data/model/proposal_session_{session_number-1}.pth"
            model.load_state_dict(torch.load(model_path, weights_only=True))
        model.train()
        new_model_path = f"../data/model/proposal_session_{session_number}.pth"

        # Initial
        if session_number == 0:
            start_time = time.time()
            clusters = initialize(model, stream, k, nbits)
            end_time = time.time()
            print(
                f"Spend {end_time-start_time} seconds for clustering({len(clusters)}, {len(stream.initial_docs)}) warming up."
            )
        # Increment
        loss_values, ts = streaming_train(ts, stream, clusters, model, lsh, num_epochs)
        write_line(
            loss_values_path, f"{session_number}, {', '.join(map(str, loss_values))}"
        )
        torch.save(model.state_dict(), new_model_path)
        # Evaluate
        clusters = evaluate_with_cluster(
            session_number, stream.docs, clusters, new_model_path, ts
        )
        # Evict
        evict_clusters(model, lsh, stream.docs, clusters, ts)
        # Pass
        prev_docs = stream.docs
        ts += 1


def evaluate_with_cluster(
    session_number, docs: dict, clusters: List[Cluster], model_path, ts, nbits=16
) -> List[Cluster]:
    eval_query_path = (
        f"/mnt/DAIS_NAS/huijeong/test_session{session_number}_queries.jsonl"
    )
    eval_queries = read_jsonl(eval_query_path, True)
    eval_docs = read_jsonl_as_dict(
        f"/mnt/DAIS_NAS/huijeong/test_session{session_number}_docs.jsonl", "doc_id"
    )
    eval_query_count, eval_doc_count = len(eval_queries), len(eval_docs)
    print(
        f"Evaluate session {session_number} | #Query:{eval_query_count}, #Document:{eval_doc_count}"
    )
    docs.update(eval_docs)
    start_time = time.time()

    # Assign
    model = BertModel.from_pretrained("bert-base-uncased").to(devices[-1])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    random_vectors = torch.randn(nbits, 768)
    lsh = RandomProjectionLSH(random_vectors=random_vectors, embedding_dim=768)
    stream_docs = list(eval_docs.values())[:3]
    assign_instance_or_add_cluster(
        model=model,
        lsh=lsh,
        clusters=clusters,
        stream_docs=stream_docs,
        docs=docs,
        ts=ts,
    )

    # Retrieve
    result = retrieve_top_k_docs_from_cluster(model, eval_queries, clusters, docs, 10)
    end_time = time.time()
    print(f"Spend {end_time-start_time} seconds for retrieval.")

    rankings_path = f"../data/rankings/proposal_{session_number}_with_cluster.txt"
    write_file(rankings_path, result)
    eval_log_path = f"../data/evals/proposal_{session_number}_with_cluster.txt"
    evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
    return clusters
