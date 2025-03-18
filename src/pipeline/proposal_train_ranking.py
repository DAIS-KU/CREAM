import random
import time
from typing import List

import torch
import pickle

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

num_gpus = 3  # torch.cuda.device_count()
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
    queries,
    docs,
    ts,
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
    use_tensor_key=False,
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
                pos_ids, pos_weights, neg_ids, neg_weights = get_samples_and_weights(
                    model=model,
                    query=query,
                    docs=docs,
                    clusters=clusters,
                    positive_k=positive_k,
                    negative_k=negative_k,
                    ts=ts,
                    use_tensor_key=use_tensor_key,
                )
                if use_label:
                    pos_docs = random.sample(query["answer_pids"], 1)[0]
                else:
                    pos_docs = [docs[_id]["text"] for _id in pos_ids]
                neg_docs = [docs[_id]["text"] for _id in neg_ids]

                query_batch.append(query["query"])
                query_embeddings = encode_texts(model=model, texts=query["query"])
                pos_embeddings = encode_texts(
                    model=model, texts=pos_docs
                )  # (positive_k, embedding_dim)
                if use_weight:
                    pos_weights = (
                        torch.tensor(pos_weights).unsqueeze(1).to(pos_embeddings.device)
                    )  # (positive_k, 1)
                    pos_embeddings = pos_embeddings * pos_weights
                pos_docs_batch.append(pos_embeddings)

                neg_embeddings = encode_texts(
                    model=model, texts=neg_docs
                )  # (negative_k, embedding_dim)
                if use_weight:
                    neg_weights = (
                        torch.tensor(neg_weights).unsqueeze(1).to(neg_embeddings.device)
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
                f"Processed {end_idx}/{query_cnt} queries | Batch Loss: {loss.item():.4f} | Total Loss: {total_loss / batch_cnt:.4f}"
            )
        end_time = time.time()
        execution_time = end_time - start_time
        total_sec += execution_time
        print(
            f"Epoch {epoch} | Total {total_sec} seconds, Avg {total_sec / batch_cnt} seconds."
        )
    return loss_values, ts


def train(
    start_session_number=0,
    end_sesison_number=3,
    load_cluster=False,
    sampling_rate=None,
    sampling_size_per_query=30,
    num_epochs=1,
    batch_size=32,
    warmingup_rate=0.2,
    negative_k=6,
    k=74,
    cluster_min_size=10,
    nbits=12,  # 16,
    max_iters=3,
    use_label=False,
    use_weight=False,
    use_tensor_key=False,
):
    ts = 0
    total_loss_values = []
    loss_values_path = "../data/loss/total_loss_values_proposal.txt"

    random_vectors = torch.randn(nbits, 768)
    lsh = RandomProjectionLSH(
        random_vectors=random_vectors, embedding_dim=768, use_tensor_key=use_tensor_key
    )
    prev_docs = None
    for session_number in range(start_session_number, end_sesison_number):
        print(f"Training Session {session_number}")
        stream = Stream(
            session_number=session_number,
            query_path=f"/mnt/DAIS_NAS/huijeong/train_session{session_number}_queries.jsonl",
            doc_path=f"/mnt/DAIS_NAS/huijeong/train_session{session_number}_docs.jsonl",
            warmingup_rate=warmingup_rate,
            sampling_rate=sampling_rate,
            sampling_size_per_query=sampling_size_per_query,
            prev_docs=prev_docs,
        )
        print(f"Session {session_number} | Document count:{len(stream.docs.keys())}")

        model = BertModel.from_pretrained("bert-base-uncased").to(devices[1])
        # if session_number != 0:
        #     model_path = f"../data/model/proposal_session_{session_number-1}.pth"
        #     model.load_state_dict(torch.load(model_path, weights_only=True))
        model.train()
        new_model_path = f"../data/model/proposal_session_{session_number}.pth"

        # Initial
        if session_number == 0:
            start_time = time.time()
            clusters = initialize(model, stream, k, nbits, max_iters)
            end_time = time.time()
            print(
                f"Spend {end_time-start_time} seconds for clustering({len(clusters)}, {len(stream.initial_docs)}) warming up."
            )
        if load_cluster:
            with open(
                f"/mnt/DAIS_NAS/huijeong/cluster_{session_number}.pkl", "rb"
            ) as f:
                pickle.dump(cluster, f)
        # # Assign
        # stream_size = stream.get_stream_size()
        # for i in range(0, stream_size):
        #     print(f"Assign {i}th stream starts.")
        #     start_time = time.time()
        #     assign_instance_or_add_cluster(
        #         model=model,
        #         lsh=lsh,
        #         clusters=clusters,
        #         cluster_min_size=cluster_min_size,
        #         stream_docs=stream.stream_docs[i],
        #         docs=stream.docs,
        #         ts=ts,
        #         use_tensor_key=use_tensor_key,
        #     )
        #     end_time = time.time()
        #     print(f"Assign {i}th stream ended({end_time - start_time}sec).")
        # # Train
        # loss_values, ts = streaming_train(
        #     queries=stream.queries,
        #     docs=stream.docs,
        #     ts=ts,
        #     clusters=clusters,
        #     model=model,
        #     lsh=lsh,
        #     num_epochs=num_epochs,
        #     negative_k=negative_k,
        #     batch_size=batch_size,
        #     use_label=use_label,
        #     use_weight=use_weight,
        # )
        # write_line(
        #     loss_values_path, f"{session_number}, {', '.join(map(str, loss_values))}"
        # )
        # torch.save(model.state_dict(), new_model_path)
        if load_cluster:
            with open(
                f"/mnt/DAIS_NAS/huijeong/cluster_{session_number}.pkl", "wb"
            ) as f:
                pickle.dump(cluster, f)
        # Evaluate
        clusters, eval_stream_docs = evaluate_with_cluster(
            session_number=session_number,
            ts=ts,
            nbits=nbits,
            clusters=clusters,
            model_path=new_model_path,
            use_tensor_key=use_tensor_key,
        )
        # Evict
        evict_clusters(model, lsh, stream.docs, clusters, ts)
        # Accumulate
        prev_docs = {**stream.docs, **eval_stream_docs}
        ts += 1


def evaluate_with_cluster(
    session_number,
    ts,
    model_path,
    clusters: List[Cluster],
    nbits,
    use_tensor_key,
) -> List[Cluster]:
    eval_query_path = (
        f"/mnt/DAIS_NAS/huijeong/test_session{session_number}_queries.jsonl"
    )
    eval_doc_path = f"/mnt/DAIS_NAS/huijeong/test_session{session_number}_docs.jsonl"
    stream = Stream(
        session_number=session_number,
        query_path=eval_query_path,
        doc_path=eval_doc_path,
    )
    eval_query_count = len(stream.queries)
    eval_doc_count = len(stream.docs)
    print(
        f"Evaluate session {session_number} | #Query:{eval_query_count}, #Document:{eval_doc_count}"
    )

    # Assign and Retrieve
    start_time = time.time()
    model = BertModel.from_pretrained("bert-base-uncased").to(devices[-1])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    result = retrieve_top_k_docs_from_cluster(
        model, stream, clusters, nbits, use_tensor_key, 10
    )
    end_time = time.time()
    print(f"Spend {end_time-start_time} seconds for retrieval.")

    rankings_path = f"../data/rankings/proposal_{session_number}_with_cluster.txt"
    write_file(rankings_path, result)
    eval_log_path = f"../data/evals/proposal_{session_number}_with_cluster.txt"
    evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
    return clusters, stream.docs
