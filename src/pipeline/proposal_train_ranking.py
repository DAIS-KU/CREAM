import torch
from transformers import BertModel, BertTokenizer
import random

from data import read_jsonl, renew_data, read_jsonl_as_dict, write_line, write_file

from functions import (
    evaluate_dataset,
    InfoNCELoss,
    show_loss,
    get_top_k_documents,
)
from cluster import (
    kmeans_pp,
    get_cluster_statistics,
    get_samples_in_clusters,
    get_samples_in_clusters_v2,
    evict_cluster_instances,
    assign_instance_or_centroid,
    update_cluster_and_current_session_data,
    process_queries_with_gpus,
)
import time
import random

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


def session_train(
    query_path,
    doc_path,
    model,
    num_epochs,
    cluster_instances,
    centroids,
    # current_session_data,
    # cluster_statistics,
    positive_k=1,
    negative_k=3,
    learning_rate=2e-5,
    batch_size=32,
    use_label=False,
):
    loss_fn = InfoNCELoss()
    learning_rate = learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    queries = read_jsonl(query_path, True)
    random.shuffle(queries)
    docs = read_jsonl_as_dict(doc_path, id_field="doc_id")
    query_cnt = len(queries)
    loss_values = []

    for epoch in range(num_epochs):
        total_loss, total_sec, batch_cnt = 0, 0, 0

        start_time = time.time()
        for start_idx in range(0, query_cnt, batch_size):
            end_idx = min(start_idx + batch_size, query_cnt)
            print(f"batch {start_idx}-{end_idx}")

            query_batch, pos_docs_batch, neg_docs_batch = [], [], []

            for qid in range(start_idx, end_idx):
                query = queries[qid]
                positive_ids, negative_ids = get_samples_in_clusters_v2(
                    model=model,
                    query=query,
                    cluster_instances=cluster_instances,
                    centroids=centroids,
                    positive_k=positive_k,
                    negative_k=negative_k,
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
    return loss_values


def train(
    sesison_count=1,
    num_epochs=1,
    batch_size=32,
    use_label=False,
    eval_cluster=True,
    negative_k=3,
):
    total_loss_values = []
    loss_values_path = "../data/loss/total_loss_values_proposal.text"
    for session_number in range(sesison_count):
        print(f"Training Session {session_number}")

        # 새로운 세션 문서
        doc_path = f"/mnt/DAIS_NAS/huijeong/train_session{session_number}_docs.jsonl"
        doc_data = read_jsonl(doc_path, False)
        _, current_session_data = renew_data(
            queries=None,
            documents=doc_data,
            nbits=16,
            embedding_dim=768,
            renew_q=False,
            renew_d=True,
        )
        print(f"Session {session_number} | Document count:{len(current_session_data)}")

        # 초기 클러스터링 구축
        if session_number == 0:
            start_time = time.time()
            centroids, cluster_instances = kmeans_pp(
                X=list(current_session_data.values()),
                k=300,
                max_iters=5,
                devices=devices,
            )
            cluster_statistics = get_cluster_statistics(
                centroids, cluster_instances, devices
            )
            end_time = time.time()
            print(
                f"Spend {end_time-start_time} seconds for clustering({len(centroids)}) warming up."
            )
        else:
            pass
        #     # 이전 세션 문서 일정량 제거된 클러스터 정보 반환
        #     centroids, cluster_instances, cluster_statistics = evict_cluster_instances(
        #         a=1.0,
        #         model=model,
        #         old_centroids=centroids,
        #         old_cluster_instances=cluster_instances,
        #         old_cluster_statistics=cluster_statistics,
        #     )
        #     # 새로운 세션 문서 클러스터 추가
        #     (centroids, cluster_statistics, cluster_instances) = (
        #         assign_instance_or_centroid(
        #             centroids=centroids,
        #             cluster_statistics=cluster_statistics,
        #             cluster_instances=cluster_instances,
        #             current_session_data=doc_data,
        #             t=session_number,
        #         )
        #     )
        # # 구/신 세션 문서 병합된 클러스터 및 현재 세션 활성화 문서들 반환
        # (cluster_instances, current_session_data) = (
        #     update_cluster_and_current_session_data(
        #         cluster_instances=cluster_instances,
        #         current_session_data=current_session_data,
        #     )
        # )

        # 샘플링 및 대조학습 수행
        model = BertModel.from_pretrained("bert-base-uncased").to(devices[-1])
        if session_number != 0:
            model_path = f"../data/model/proposal_session_{session_number-1}.pth"
            model.load_state_dict(torch.load(model_path))
        model.train()
        new_model_path = f"../data/model/proposal_session_{session_number}.pth"

        loss_values = session_train(
            query_path=f"/mnt/DAIS_NAS/huijeong/train_session{session_number}_queries.jsonl",
            doc_path=doc_path,
            model=model,
            num_epochs=num_epochs,
            batch_size=batch_size,
            centroids=centroids,
            cluster_instances=cluster_instances,
            # current_session_data=current_session_data,
            negative_k=negative_k,
        )
        # total_loss_values.extend(loss_values)
        write_line(
            loss_values_path, f"{session_number}, {', '.join(map(str, loss_values))}"
        )
        torch.save(model.state_dict(), new_model_path)
        if eval_cluster:
            evaluate_with_cluster(
                session_number,
                centroids,
                cluster_statistics,
                cluster_instances,
                new_model_path,
            )
        else:
            evaluate_without_cluster(session_number, new_model_path)


def evaluate_with_cluster(
    session_number, centroids, cluster_statistics, cluster_instances, model_path
):
    print(f"Evaluate session {session_number} with cluster")
    eval_query_path = (
        f"/mnt/DAIS_NAS/huijeong/test_session{session_number}_queries.jsonl"
    )
    eval_doc_path = f"/mnt/DAIS_NAS/huijeong/test_session{session_number}_docs.jsonl"

    eval_query_data = read_jsonl(eval_query_path, True)
    eval_doc_data = read_jsonl(eval_doc_path, False)

    eval_query_count = len(eval_query_data)
    eval_doc_count = len(eval_doc_data)
    print(f"Query count:{eval_query_count}, Document count:{eval_doc_count}")

    new_q_data, new_d_data = renew_data(
        queries=eval_query_data,
        documents=eval_doc_data,
        nbits=0,
        embedding_dim=768,
        model_path=model_path,
        renew_q=True,
        renew_d=True,
    )
    # Assign test docs
    (centroids, cluster_statistics, cluster_instances) = assign_instance_or_centroid(
        centroids=centroids,
        cluster_statistics=cluster_statistics,
        cluster_instances=cluster_instances,
        current_session_data=new_d_data,
        t=session_number,
        devices=devices,
    )
    # Retrieve test queries
    start_time = time.time()
    result = process_queries_with_gpus(
        new_q_data, centroids, cluster_instances, devices
    )
    end_time = time.time()
    print(f"Spend {end_time-start_time} seconds for retrieval.")

    rankings_path = f"../data/rankings/proposal_{session_number}_with_cluster.txt"
    write_file(rankings_path, result)
    eval_log_path = f"../data/evals/proposal_{session_number}_with_cluster.txt"
    evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
    del new_q_data, new_d_data
    # TODO Return updated cluster?


def evaluate_without_cluster(session_number, model_path):
    print(f"Evaluate session {session_number} without cluster")
    eval_query_path = (
        f"/mnt/DAIS_NAS/huijeong/test_session{session_number}_queries.jsonl"
    )
    eval_doc_path = f"/mnt/DAIS_NAS/huijeong/test_session{session_number}_docs.jsonl"

    eval_query_data = read_jsonl(eval_query_path, True)
    eval_doc_data = read_jsonl(eval_doc_path, False)

    eval_query_count = len(eval_query_data)
    eval_doc_count = len(eval_doc_data)
    print(f"Query count:{eval_query_count}, Document count:{eval_doc_count}")

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

    rankings_path = f"../data/rankings/proposal_{session_number}_wo_cluster.txt"
    write_file(rankings_path, result)
    eval_log_path = f"../data/evals/proposal_{session_number}_wo_cluster.txt"
    evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
    del new_q_data, new_d_data
