import random
import torch
from transformers import BertModel, BertTokenizer

from functions import get_samples_in_clusters
from cluster import kmeans_pp

torch.autograd.set_detect_anomaly(True)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def encode_texts(model, texts, device, max_length=256, is_cls=True):
    no_padding_inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    no_padding_inputs = {
        key: value.to(device) for key, value in no_padding_inputs.items()
    }
    outputs = model(**no_padding_inputs).last_hidden_state
    if is_cls:
        embedding = outputs[:, 0, :]  # [CLS]만 사용
    else:
        embedding = outputs[0, 1:-1, :]  # [CLS]와 [SEP] 제외
    return embedding


def session_train(
    query_path,
    model,
    num_epochs,
    batch_size,
    cluster_instances,
    best_centroids,
    current_session_data,
    centroids_statics,
    loss_fn,
    optimizer,
    positive_k,
    negative_k,
):
    queries = load_jsonl(query_path)
    query_cnt = len(queries)
    loss_values = []
    centroid_lsh_tensor = torch.stack(best_centroids)

    for epoch in range(num_epochs):
        total_loss = 0
        lack_of_positive_samples, lack_of_negative_samples, lack_of_sample_queries = (
            0,
            0,
            0,
        )
        # print_cluster_instances(1.0, model, best_centroids, cluster_instances, centroids_statics)

        for start_idx in range(0, query_cnt, batch_size):
            end_idx = min(start_idx + batch_size, query_cnt)
            print(f"batch {start_idx}-{end_idx}")

            query_batch, pos_docs_batch, neg_docs_batch = [], [], []

            for qid in range(start_idx, end_idx):
                query = queries[qid]
                positive_ids, negative_ids = get_samples_in_clusters(
                    model=model,
                    query=query,
                    cluster_instances=cluster_instances,
                    centroid_lsh_tensor=centroid_lsh_tensor,
                    positive_k=positive_k,
                    negative_k=negative_k,
                )
                if len(positive_ids) < positive_k:
                    lack_of_positive_samples += 1
                if len(negative_ids) < negative_k:
                    lack_of_negative_samples += 1
                if len(positive_ids) < positive_k or len(negative_ids) < negative_k:
                    lack_of_sample_queries += 1

                pos_docs = [current_session_data[_id]["TEXT"] for _id in positive_ids]
                neg_docs = [current_session_data[_id]["TEXT"] for _id in negative_ids]

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
            print(
                f"Lack of positives: {lack_of_positive_samples}, Lack of negatives : {lack_of_negative_samples} for queries {lack_of_sample_queries}"
            )
    return loss_values


def train(sesison_count=1, num_epochs=1, batch_size=32, positive_k=1, negative_k=3):
    for session_number in range(sesison_count):
        print(f"Running Session {session_number}")

        # 새로운 세션 문서
        doc_path = f"../data/sessions/train_session{session_number}_docs.jsonl"
        doc_data = read_jsonl(doc_path)
        _, doc_data = renew_data(None, doc_data, 24, 768)
        print(f"Session {session_number} | Document count:{len(doc_data)}")

        # 초기 클러스터링 구축
        if session_number == 0:
            centroids, labels, centroids_statics = kmeans_pp(
                list(doc_data.values()), 10, 10, devices
            )
        else:
            # 이전 세션 문서 일정량 제거된 클러스터 정보 반환
            centroids, cluster_instances, centroids_statics = evict_cluster_instances(
                a=1.0,
                model=model,
                old_centroids=centroids,
                old_cluster_instances=cluster_instances,
                old_centroids_statics=centroids_statics,
            )
            # 새로운 세션 문서 클러스터 추가
            centroids, centroids_statics, cluster_instances, labels = (
                assign_instance_or_centroid(
                    centroids=centroids,
                    centroids_statics=centroids_statics,
                    cluster_instances=cluster_instances,
                    current_session_data=current_session_data,
                    t=t,
                )
            )
        # 구/신 세션 문서 병합된 클러스터 및 현재 세션 활성화 문서들 반환
        cluster_instances, current_session_data, labels = (
            update_cluster_and_current_session_data(
                cluster_instances=cluster_instances,
                current_session_data=current_session_data,
                labels=labels,
            )
        )

        # 샘플링 및 대조학습 수행
        loss_values = train(
            query_path=f"./raw/sessions/train_session{session_number}_queries.jsonl",
            model=model,
            num_epochs=num_epochs,
            batch_size=batch_size,
            cluster_instances=cluster_instances,
            best_centroids=centroids,
            current_session_data=current_session_data,
            loss_fn=loss_fn,
            optimizer=optimizer,
            positive_k=positive_k,
            negative_k=negative_k,
        )
        total_loss_values.append(loss_values)
        show_loss(total_loss_values)

        model_path = f"./model/proposal_session_{session_number}.pth"
        torch.save(model.state_dict(), model_path)
