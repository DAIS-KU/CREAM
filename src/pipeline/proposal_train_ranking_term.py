import random
import time
from typing import List
import numpy as np

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
    clear_invalid_clusters,
    renew_data,
    make_query_psuedo_answers,
)
from data import Stream, read_jsonl, read_jsonl_as_dict, write_file, write_line
from functions import (
    InfoNCELoss,
    evaluate_dataset,
    get_top_k_documents,
    InfoNCETermLoss,
)

torch.autograd.set_detect_anomaly(True)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

num_gpus = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
print(devices)

# def encode_texts(model, texts, max_length=256):
#     device = model.device
#     no_padding_inputs = tokenizer(
#         texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
#     )
#     no_padding_inputs = {
#         key: value.to(device) for key, value in no_padding_inputs.items()
#     }
#     outputs = model(**no_padding_inputs).last_hidden_state
#     embedding = outputs[:, 0, :]  # [CLS]만 사용
#     return embedding

# def encode_texts(model, texts, max_length=256):
#     device = model.device
#     batch_inputs = tokenizer(
#         texts,
#         return_tensors="pt",
#         truncation=True,
#         padding="max_length",
#         max_length=max_length,
#     )
#     batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
#     outputs = model(**batch_inputs).last_hidden_state
#     attention_mask = batch_inputs["attention_mask"].unsqueeze(
#         -1
#     )
#     token_embeddings = outputs[
#         :, :max_length, :
#     ]
#     masked_embeddings = token_embeddings * attention_mask
#     mean_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
#     return mean_embeddings


def encode_texts(model, texts, max_length=256):
    device = model.device
    batch_inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
    outputs = model(**batch_inputs).last_hidden_state
    token_embeddings = outputs[:, 1:-1, :]
    attention_mask = batch_inputs["attention_mask"][:, 1:-1]
    token_embeddings = token_embeddings * (attention_mask[:, :, None].to(device))
    return token_embeddings


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
    loss_fn = InfoNCETermLoss()
    learning_rate = learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_values = []

    query_answers = make_query_psuedo_answers(
        model, queries, docs, clusters, use_tensor_key
    )
    all_answer_ids = set(query_answers.values())

    for epoch in range(num_epochs):
        total_loss, total_sec, batch_cnt = 0, 0, 0

        start_time = time.time()
        for start_idx in range(0, query_cnt, batch_size):
            end_idx = min(start_idx + batch_size, query_cnt)
            print(f"query {start_idx}-{end_idx}")

            query_batch, pos_docs_batch, neg_docs_batch = [], [], []
            for idx in range(start_idx, end_idx):
                query = queries[idx]
                # pos_ids, pos_weights, neg_ids, neg_weights = get_samples_and_weights(
                # pos_ids, neg_ids, _ = get_samples_and_weights(
                #     model=model,
                #     query=query,
                #     docs=docs,
                #     clusters=clusters,
                #     positive_k=positive_k,
                #     negative_k=negative_k,
                #     ts=ts,
                #     use_tensor_key=use_tensor_key,
                # )

                pos_ids = [query_answers[query["qid"]]]
                candidate_neg_ids = list(all_answer_ids - set(pos_ids))
                neg_ids = random.sample(
                    candidate_neg_ids, min(6, len(candidate_neg_ids))
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
                # if use_weight:
                #     pos_weights = (
                #         torch.tensor(pos_weights).unsqueeze(1).to(pos_embeddings.device)
                #     )  # (positive_k, 1)
                #     pos_embeddings = pos_embeddings * pos_weights
                pos_docs_batch.append(pos_embeddings)

                neg_embeddings = encode_texts(
                    model=model, texts=neg_docs
                )  # (negative_k, embedding_dim)
                # if use_weight:
                #     neg_weights = (
                #         torch.tensor(neg_weights).unsqueeze(1).to(neg_embeddings.device)
                #     )  # (negative_k, 1)
                #     neg_embeddings = neg_embeddings * neg_weights
                neg_docs_batch.append(neg_embeddings)

            query_embeddings = encode_texts(
                model=model, texts=query_batch
            )  # (batch_size, embedding_dim)
            positive_embeddings = torch.stack(
                pos_docs_batch
            )  # (batch_size, positive_k, embedding_dim)
            negative_embeddings = torch.stack(neg_docs_batch)
            d_embeddings = torch.cat((positive_embeddings, negative_embeddings), dim=1)
            # print(f"d_embeddings: {d_embeddings.shape}")
            loss = loss_fn(query_embeddings, d_embeddings)

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
    positive_k=1,
    negative_k=6,
    cluster_min_size=10,
    nbits=12,
    max_iters=3,
    init_k=None,
    use_label=False,
    use_weight=False,
    use_tensor_key=False,
    warming_up_method=None,
):
    total_loss_values = []
    loss_values_path = "../data/loss/total_loss_values_proposal_term.txt"
    required_doc_size = positive_k + negative_k  # 임의 설정하면(최대, 최소 갯수 생기면) 배치 안 됨

    random_vectors = torch.randn(nbits, 768)
    lsh = RandomProjectionLSH(
        random_vectors=random_vectors, embedding_dim=768, use_tensor_key=use_tensor_key
    )
    prev_docs, clusters = None, None
    for session_number in range(start_session_number, end_sesison_number):
        ts = start_session_number
        print(f"Training Session {session_number}")
        stream = Stream(
            session_number=session_number,
            query_path=f"../data/sub/train_session{session_number}_queries.jsonl",
            doc_path=f"../data/sub/train_session{session_number}_docs.jsonl",
            warmingup_rate=warmingup_rate,
            sampling_rate=sampling_rate,
            prev_docs=prev_docs,
            sampling_size_per_query=sampling_size_per_query,
            warming_up_method=warming_up_method,
        )
        print(f"Session {session_number} | Document count:{len(stream.docs.keys())}")

        model = BertModel.from_pretrained("bert-base-uncased").to(devices[1])
        if session_number != 0:
            model_path = f"../data/model/proposal_session_{session_number-1}_term.pth"
        else:
            print("Load Warming up model.")
            model_path = f"../data/base_model_msmarco.pth"
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.train()
        new_model_path = f"../data/model/proposal_session_{session_number}_term.pth"

        # Initial
        if load_cluster and session_number > 0:
            with open(f"../data/clusters_{session_number-1}_term.pkl", "rb") as f:
                clusters = pickle.load(f)
            with open(f"../data/prev_docs_{session_number-1}_term.pkl", "rb") as f:
                prev_docs = pickle.load(f)
            with open(f"../data/random_vectors_{session_number-1}_term.pkl", "rb") as f:
                random_vectors = pickle.load(f)
            stream.docs.update(prev_docs)
            batch_start = 0
        else:
            if session_number == 0:
                start_time = time.time()
                if warming_up_method == "initial_cluster":
                    init_k = (
                        int(np.log2(len(stream.initial_docs)))
                        if init_k is None
                        else init_k
                    )
                    clusters = initialize(
                        model,
                        stream.initial_docs,
                        stream.docs,
                        init_k,
                        nbits,
                        max_iters,
                        use_tensor_key,
                    )
                    initial_size = len(stream.initial_docs)
                    batch_start = 0
                elif warming_up_method == "query_seed":
                    init_k = (
                        int(np.log2(len(stream.stream_queries[0])))
                        if init_k is None
                        else init_k
                    )
                    clusters = initialize(
                        model,
                        stream.stream_queries[0],
                        stream.docs,
                        init_k,
                        nbits,
                        max_iters,
                        use_tensor_key,
                    )
                    initial_size = len(stream.stream_queries[0])
                    batch_start = 1
                elif warming_up_method == "stream_seed":
                    init_k = (
                        int(np.log2(len(stream.stream_docs[0])))
                        if init_k is None
                        else init_k
                    )
                    clusters = initialize(
                        model,
                        stream.stream_docs[0],
                        stream.docs,
                        init_k,
                        nbits,
                        max_iters,
                        use_tensor_key,
                    )
                    initial_size = len(stream.stream_docs[0])
                    batch_start = 1
                elif warming_up_method == "none":
                    batch_start = 0
                else:
                    raise NotImplementedError(
                        f"Unsupported warming_up_method: {warming_up_method}"
                    )
                end_time = time.time()
                print(
                    f"Spend {end_time-start_time} seconds for clustering({len(clusters)}, {initial_size}) warming up."
                )
            else:
                raise ValueError("Loading or initialization is required.")

        # Assign stream batch
        for i in range(batch_start, len(stream.stream_docs)):
            print(f"Assign {i}th stream starts.")
            start_time = time.time()
            assign_instance_or_add_cluster(
                model=model,
                lsh=lsh,
                clusters=clusters,
                cluster_min_size=cluster_min_size,
                stream_docs=stream.stream_docs[i],
                docs=stream.docs,
                ts=ts,
                use_tensor_key=use_tensor_key,
            )
            if i % 50 == 0:
                for j, cluster in enumerate(clusters):
                    print(f"{j}th size: {len(cluster.doc_ids)}")
        end_time = time.time()
        print(f"Assign {i}th stream ended({end_time - start_time}sec).")

        # Remain only trainable clusters
        clusters = clear_invalid_clusters(clusters, stream.docs, required_doc_size)
        # Train
        loss_values, ts = streaming_train(
            queries=stream.queries,
            docs=stream.docs,
            ts=ts,
            clusters=clusters,
            model=model,
            lsh=lsh,
            num_epochs=num_epochs,
            negative_k=negative_k,
            batch_size=batch_size,
            use_label=use_label,
            use_weight=use_weight,
        )
        write_line(
            loss_values_path, f"{session_number}, {', '.join(map(str, loss_values))}"
        )
        torch.save(model.state_dict(), new_model_path)

        # Evaluate
        clusters, eval_stream_docs = evaluate_with_cluster(
            session_number=session_number,
            ts=ts,
            random_vectors=random_vectors,
            clusters=clusters,
            model_path=new_model_path,
            use_tensor_key=use_tensor_key,
        )
        # Evict
        evict_clusters(model, lsh, stream.docs, clusters, ts, required_doc_size)
        # Accumulate
        prev_docs = {**stream.docs, **eval_stream_docs}
        if load_cluster:
            with open(f"../data/clusters_{session_number}_term.pkl", "wb") as f:
                pickle.dump(clusters, f)
            with open(f"../data/prev_docs_{session_number}_term.pkl", "wb") as f:
                pickle.dump(prev_docs, f)
            with open(f"../data/random_vectors_{session_number}_term.pkl", "wb") as f:
                pickle.dump(random_vectors, f)


def evaluate_with_cluster(
    session_number,
    ts,
    model_path,
    clusters: List[Cluster],
    random_vectors,
    use_tensor_key,
) -> List[Cluster]:
    eval_query_path = f"../data/sub/test_session{session_number}_queries.jsonl"
    eval_doc_path = f"../data/sub/test_session{session_number}_docs.jsonl"
    stream = Stream(
        session_number=session_number,
        query_path=eval_query_path,
        doc_path=eval_doc_path,
        warming_up_method="eval",
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
        model, stream, clusters, random_vectors, use_tensor_key, 10
    )
    end_time = time.time()
    print(f"Spend {end_time-start_time} seconds for retrieval.")

    rankings_path = f"../data/rankings/proposal_{session_number}_with_cluster_term.txt"
    write_file(rankings_path, result)
    eval_log_path = f"../data/evals/proposal_{session_number}_with_cluster_term.txt"
    evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
    return clusters, stream.docs


def evaluate(session_number):
    method = "proposal"
    print(f"Evaluate Session {session_number}")
    eval_query_path = f"../data/sub/test_session{session_number}_queries.jsonl"
    eval_doc_path = f"../data/sub/test_session{session_number}_docs.jsonl"

    eval_query_data = read_jsonl(eval_query_path, True)
    eval_doc_data = read_jsonl(eval_doc_path, False)

    eval_query_count = len(eval_query_data)
    eval_doc_count = len(eval_doc_data)
    print(f"Query count:{eval_query_count}, Document count:{eval_doc_count}")

    rankings_path = f"../data/rankings/{method}_session_{session_number}_term.txt"
    model_path = f"../data/model/{method}_session_{session_number}_term.pth"

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

    rankings_path = f"../data/rankings/{method}_session_{session_number}_term.txt"
    write_file(rankings_path, result)
    eval_log_path = f"../data/evals/{method}_{session_number}_term.txt"
    evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
    del new_q_data, new_d_data
