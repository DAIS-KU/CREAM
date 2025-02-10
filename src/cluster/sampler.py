from transformers import BertModel, BertTokenizer
import torch
from functions import calculate_S_qd_regl_dict, calculate_S_qd_regl_batch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def get_passage_embeddings(model, passages, max_length=256):
    device = model.device
    batch_inputs = tokenizer(
        passages,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
    with torch.no_grad():
        outputs = model(**batch_inputs).last_hidden_state
    token_embeddings = outputs[:, 1:-1, :]
    attention_mask = batch_inputs["attention_mask"][:, 1:-1]
    token_embeddings = token_embeddings * (attention_mask[:, :, None].to(device))
    return token_embeddings


def get_samples_in_clusters(
    model, query, cluster_instances, centroids, positive_k, negative_k
):
    device = model.device
    query_token_embs = get_passage_embeddings(model, query["query"])
    distances = []
    for centroid in centroids:
        distances.append(calculate_S_qd_regl_dict(query_token_embs, centroid, device))
    _, indices = torch.topk(
        torch.stack(distances), k=3, largest=False
    )  # TODO 거리대신 점수 비교로 수정
    positive_id, negative_id, negative_2nd_id = (
        indices[0].item(),
        indices[1].item(),
        indices[2].item(),
    )
    print(f"positive_id:{positive_id} | negative_id:{negative_id}")

    positive_samples = []
    pos_docs = cluster_instances[positive_id]
    available_pos_doc_count = len(pos_docs)
    # print(f'available_pos_doc_count: {available_pos_doc_count}')
    if available_pos_doc_count != 0:
        pos_doc_tensor = torch.stack([doc["TOKEN_EMBS"] for doc in pos_docs])
        pos_doc_scores = calculate_S_qd_regl_batch(
            query_token_embs, pos_doc_tensor, device
        )
        # print(f'pos_doc_scores: {pos_doc_scores.shape}')
        _, pos_doc_top_k_indices = torch.topk(
            pos_doc_scores, k=min(positive_k, available_pos_doc_count)
        )
        # print(f'pos_doc_top_k_indices.tolist(): {pos_doc_top_k_indices.tolist()}')
        positive_samples.extend(
            [pos_docs[pidx]["ID"] for pidx in pos_doc_top_k_indices.tolist()]
        )

    negative_samples = []
    neg_docs = cluster_instances[negative_id]
    available_neg_doc_count = len(neg_docs)
    if available_neg_doc_count != 0:
        neg_doc_tensor = torch.stack([doc["TOKEN_EMBS"] for doc in neg_docs])
        neg_doc_scores = calculate_S_qd_regl_batch(
            query_token_embs, neg_doc_tensor, device
        )
        # print(f'neg_doc_scores: {neg_doc_scores.shape}')
        _, neg_bottom_k_indices = torch.topk(
            neg_doc_scores, k=min(negative_k, available_neg_doc_count), largest=False
        )
        # print(f'neg_bottom_k_indices.tolist(): {neg_bottom_k_indices.tolist()}')
        negative_samples.extend(
            [neg_docs[nidx]["ID"] for nidx in neg_bottom_k_indices.tolist()]
        )

    # 모자라는 만큼 다음 클러스터에서 가져옴
    if available_neg_doc_count < negative_k:
        # print('available_neg_doc_count < negative_k')
        need_k = negative_k - available_neg_doc_count
        neg_docs = cluster_instances[negative_2nd_id]
        if len(neg_docs) < need_k:
            print(f"[WARN] neg_docs_2nd < need_k !")
        neg_doc_tensor = torch.stack([doc["TOKEN_EMBS"] for doc in neg_docs])
        neg_doc_scores = calculate_S_qd_regl_batch(
            query_token_embs, neg_doc_tensor, device
        )
        # print(f'neg_doc_scores: {neg_doc_scores.shape}')
        _, neg_bottom_k_indices = torch.topk(
            neg_doc_scores, k=min(len(neg_docs), need_k), largest=False
        )
        # print(f'neg_bottom_k_indices.tolist(): {neg_bottom_k_indices.tolist()}')
        negative_samples.extend(
            [neg_docs[nidx]["ID"] for nidx in neg_bottom_k_indices.tolist()]
        )

    print(
        f" query: {query['qid']} | positive: {positive_samples} | negative:{negative_samples}"
    )
    return positive_samples, negative_samples


def get_samples_in_clusters_v2(
    model, query, cluster_instances, centroids, positive_k=1, negative_k=3
):
    device = model.device
    query_token_embs = get_passage_embeddings(model, query["query"])
    distances = []

    for centroid in centroids:
        distances.append(calculate_S_qd_regl_dict(query_token_embs, centroid, device))

    # 거리 기반으로 정렬 (가장 가까운 1개, 가장 먼 k개)
    sorted_distances, sorted_indices = torch.sort(
        torch.stack(distances), descending=False
    )

    positive_id = sorted_indices[0].item()  # 가장 가까운 클러스터 (positive)
    negative_ids = sorted_indices[
        -negative_k:
    ].tolist()  # 가장 먼 k개 클러스터 (negative)

    print(f"positive_id:{positive_id} | negative_ids:{negative_ids}")

    # positive sample: 가장 가까운 클러스터에서 top-1 문서 선택
    positive_samples = []
    pos_docs = cluster_instances[positive_id]
    if len(pos_docs) != 0:
        pos_doc_tensor = torch.stack([doc["TOKEN_EMBS"] for doc in pos_docs])
        pos_doc_scores = calculate_S_qd_regl_batch(
            query_token_embs, pos_doc_tensor, device
        )
        _, pos_doc_top_k_indices = torch.topk(pos_doc_scores, k=1)  # top-1 문서 선택
        positive_samples.extend(
            [pos_docs[pidx]["ID"] for pidx in pos_doc_top_k_indices.tolist()]
        )

    # negative samples: 가장 먼 클러스터들에서 각각 top-1 문서 선택
    negative_samples = []
    for neg_id in negative_ids:
        neg_docs = cluster_instances[neg_id]
        if len(neg_docs) != 0:
            neg_doc_tensor = torch.stack([doc["TOKEN_EMBS"] for doc in neg_docs])
            neg_doc_scores = calculate_S_qd_regl_batch(
                query_token_embs, neg_doc_tensor, device
            )
            _, neg_bottom_k_indices = torch.topk(
                neg_doc_scores, k=1, largest=False
            )  # top-1 문서 선택
            negative_samples.extend(
                [neg_docs[nidx]["ID"] for nidx in neg_bottom_k_indices.tolist()]
            )

    print(
        f" query: {query['qid']} | positive: {positive_samples} | negative:{negative_samples}"
    )
    return positive_samples, negative_samples
