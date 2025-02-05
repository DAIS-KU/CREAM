def get_samples_in_clusters(
    model, query, cluster_instances, centroid_lsh_tensor, positive_k, negative_k
):
    query_token_embs = get_passage_embeddings(model, query["query"])
    distances = 1.0 - calculate_S_qd_regl_batch(query_token_embs, centroid_lsh_tensor)
    _, indices = torch.topk(
        distances, k=3, largest=False
    )  # 반환값 다 텐서, 거리이므로 최단거리
    positive_id, negative_id, negative_2nd_id = (
        indices[0].item(),
        indices[1].item(),
        indices[2].item(),
    )
    print(f"positive_id:{positive_id} | negative_id:{negative_id}")

    positive_samples = []
    pos_docs = [
        pos_doc
        for pos_doc in cluster_instances[positive_id]
        if pos_doc["TYPE"] == "document"
    ]
    available_pos_doc_count = len(pos_docs)
    if available_pos_doc_count != 0:
        pos_doc_lsh_tensor = torch.stack(
            [get_lsh_vector(model, doc["TEXT"]) for doc in pos_docs]
        )
        pos_doc_scores = calculate_S_qd_regl_batch(query_token_embs, pos_doc_lsh_tensor)
        _, pos_doc_top_k_indices = torch.topk(
            pos_doc_scores, k=min(positive_k, available_pos_doc_count)
        )
        positive_samples.extend(
            [pos_docs[pidx]["ID"] for pidx in pos_doc_top_k_indices.tolist()]
        )

    # 모자라는 만큼 쿼리에서 가져옴
    if available_pos_doc_count < positive_k:
        need_k = positive_k - available_pos_doc_count
        pos_queries = [
            pos_query
            for pos_query in cluster_instances[positive_id]
            if pos_query["TYPE"] == "query" and pos_query["ID"] != query["qid"]
        ]
        # Withes-Pentaur-Therm-Libation-Tickled-Pp
        if len(pos_queries) == 0:
            pos_queries = [pos_query for pos_query in cluster_instances[positive_id]]
        pos_queries_lsh_tensor = torch.stack(
            [get_lsh_vector(model, q["TEXT"]) for q in pos_queries]
        )
        pos_queries_scores = calculate_S_qd_regl_batch(
            query_token_embs, pos_queries_lsh_tensor
        )
        _, pos_query_top_k_indices = torch.topk(pos_queries_scores, k=need_k)
        # print(f'pos_query_top_k_indices:{pos_query_top_k_indices.tolist()}')
        positive_samples.extend(
            [pos_queries[pidx]["ID"] for pidx in pos_query_top_k_indices.tolist()]
        )

    negative_samples = []
    neg_docs = cluster_instances[negative_id]
    available_neg_doc_count = len(neg_docs)
    if available_neg_doc_count != 0:
        neg_doc_lsh_tensor = torch.stack(
            [get_lsh_vector(model, doc["TEXT"]) for doc in neg_docs]
        )
        neg_doc_scores = calculate_S_qd_regl_batch(query_token_embs, neg_doc_lsh_tensor)
        _, neg_bottom_k_indices = torch.topk(
            neg_doc_scores, k=min(negative_k, available_neg_doc_count), largest=False
        )
        negative_samples.extend(
            [neg_docs[nidx]["ID"] for nidx in neg_bottom_k_indices.tolist()]
        )

    # 모자라는 만큼 다음 클러스터에서 가져옴
    if available_neg_doc_count < negative_k:
        need_k = negative_k - available_neg_doc_count
        neg_docs = cluster_instances[negative_2nd_id]
        neg_doc_lsh_tensor = torch.stack(
            [get_lsh_vector(model, neg_doc["TEXT"]) for neg_doc in neg_docs]
        )
        neg_doc_scores = calculate_S_qd_regl_batch(query_token_embs, neg_doc_lsh_tensor)
        _, neg_bottom_k_indices = torch.topk(neg_doc_scores, k=need_k, largest=False)
        negative_samples.extend(
            [neg_docs[nidx]["ID"] for nidx in neg_bottom_k_indices.tolist()]
        )

    print(
        f" query: {query['qid']} | positive: {positive_samples} | negative:{negative_samples}"
    )
    return positive_samples, negative_samples


def rehearsal_sampling(query, method):
    if method == "ER":
        pass
    elif method == "MIR":
        pass
    elif method == "GSS":
        pass
    elif method == "OCS":
        pass
    elif method == "L2R":
        pass
    else:
        raise ValueError(f"Unsupported rehearsal method {method}")
