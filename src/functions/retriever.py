import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

from functions import calculate_S_qd_regl_batch


def get_top_k_documents_by_cosine(query_emb, current_data_embs, k=10):
    scores = torch.nn.functional.cosine_similarity(
        query_emb.unsqueeze(0), current_data_embs, dim=1
    )
    top_k_scores, top_k_indices = torch.topk(scores, k)
    return top_k_indices.tolist()


def get_top_k_documents_gpu(new_q_data, docs, query_id, k, devices, batch_size=2048):
    query_data_item = new_q_data[query_id]
    query_token_embs = query_data_item["TOKEN_EMBS"]

    docs_cnt = len(docs)
    num_gpus = len(devices)
    batch_indices = [
        list(range(i * batch_size, min((i + 1) * batch_size, docs_cnt)))
        for i in range((docs_cnt + batch_size - 1) // batch_size)
    ]
    gpu_batch_indices = [batch_indices[i::num_gpus] for i in range(num_gpus)]

    def process(query_token_embs, gpu_batch_indices, device):
        query_token_embs = query_token_embs.unsqueeze(0).to(device)
        regl_scores = []
        for batch in gpu_batch_indices:
            start_idx, end_idx = batch[0], batch[-1] + 1
            print(f"{device}| Processing batch {start_idx}-{end_idx}")
            combined_embs = torch.stack(
                [doc["TOKEN_EMBS"].to(device) for doc in docs[start_idx:end_idx]], dim=0
            )
            # print(f'query_token_embs:{query_token_embs.shape}, combined_embs:{combined_embs.shape}')
            regl_score = calculate_S_qd_regl_batch(
                query_token_embs, combined_embs, device
            )
            regl_scores.extend(
                [
                    (doc["ID"], regl_score[idx].item())
                    for idx, doc in enumerate(docs[start_idx:end_idx])
                ]
            )
        return regl_scores

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        args = [
            (query_token_embs, gpu_batch_indices[i], devices[i])
            for i in range(num_gpus)
        ]
        results = list(executor.map(lambda p: process(*p), args))

    combined_regl_scores = [score for gpu_scores in results for score in gpu_scores]
    combined_regl_scores = sorted(
        combined_regl_scores, key=lambda x: x[1], reverse=True
    )
    top_k_regl_docs = combined_regl_scores[:k]
    top_k_regl_doc_ids = [x[0] for x in top_k_regl_docs]

    return top_k_regl_doc_ids


def get_top_k_documents(new_q_data, new_d_data, k=10, batch_size=2048):
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    print(f"Using GPUs: {devices}")

    docs = list(new_d_data.values())
    results = {}
    for qcnt, query_id in enumerate(new_q_data.keys()):
        results[query_id] = get_top_k_documents_gpu(
            new_q_data, docs, query_id, k, devices, batch_size
        )
        if qcnt % 10 == 0:
            print(f"#{qcnt} retrieving is done.")
    return results
