from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import torch
from transformers import BertTokenizer

from data import BM25Okapi
from functions import process_batch
import torch.nn.functional as F
from clusters import (
    renew_data,
    calculate_S_qd_regl_batch,
    calculate_S_qd_regl_batch_batch,
)

tokenizer = BertTokenizer.from_pretrained("/home/work/.default/huijeong/bert_local")


def preprocess(corpus, max_length=256):
    # corpus = corpus.lower()
    # corpus = "".join([char for char in corpus if char not in string.punctuation])
    # tokens = word_tokenize(corpus)
    # max_len = min(max_length, len(tokens))
    # tokens = tokens[:max_len]
    tokens = corpus.split(" ")
    return tokens


def encode_data_mean_pooling(model, queries_data, documents_data):
    num_gpus = torch.cuda.device_count()
    models = [deepcopy(model) for _ in range(num_gpus)]
    query_batches = []
    doc_batches = []

    for i in range(num_gpus):
        query_start = i * len(queries_data) // num_gpus
        query_end = (i + 1) * len(queries_data) // num_gpus
        query_batches.append(queries_data[query_start:query_end])

        doc_start = i * len(documents_data) // num_gpus
        doc_end = (i + 1) * len(documents_data) // num_gpus
        doc_batches.append(documents_data[doc_start:doc_end])

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        print(f"Query mean-pooling embedding starts.")
        futures = []
        for i in range(num_gpus):
            if query_batches[i]:
                futures.append(
                    executor.submit(
                        process_batch,
                        query_batches[i],
                        models[i],
                        tokenizer,
                        i,
                        "qid",
                        "query",
                    )
                )
        query_results = {}
        for future in futures:
            query_results.update(future.result())

        print(f"Document mean-pooling embedding starts..")
        futures = []
        for i in range(num_gpus):
            if doc_batches[i]:
                futures.append(
                    executor.submit(
                        process_batch,
                        doc_batches[i],
                        models[i],
                        tokenizer,
                        i,
                        "doc_id",
                        "text",
                    )
                )
        doc_results = {}
        for future in futures:
            doc_results.update(future.result())
    return query_results, doc_results


def get_top_k_documents_cosine(query_emb, docs, k, devices, batch_size=4096):
    docs_cnt = len(docs)
    num_gpus = len(devices)
    batch_indices = [
        list(range(i * batch_size, min((i + 1) * batch_size, docs_cnt)))
        for i in range((docs_cnt + batch_size - 1) // batch_size)
    ]
    gpu_batch_indices = [batch_indices[i::num_gpus] for i in range(num_gpus)]

    def process_cosine(query_emb, gpu_batch_indices, device):
        query_emb = query_emb.clone().to(device)
        scores = []
        for batch in gpu_batch_indices:
            start_idx, end_idx = batch[0], batch[-1] + 1
            print(f"{device}| Processing batch {start_idx}-{end_idx}")
            combined_embs = torch.stack(
                [doc["EMB"] for doc in docs[start_idx:end_idx]], dim=0
            ).to(device)
            score = F.cosine_similarity(
                # query_emb.unsqueeze(1), combined_embs.unsqueeze(0), dim=-1
                query_emb.unsqueeze(0),
                combined_embs,
                dim=-1,
            ).squeeze()
            # print(f"score: {score.shape}")
            scores.extend(
                [
                    (doc["doc_id"], score[idx].item())
                    for idx, doc in enumerate(docs[start_idx:end_idx])
                ]
            )
        return scores

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        args = [(query_emb, gpu_batch_indices[i], devices[i]) for i in range(num_gpus)]
        results = list(executor.map(lambda p: process_cosine(*p), args))

    combined_scores = [score for gpu_scores in results for score in gpu_scores]
    combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)
    top_k_docs, bottom_k_docs = combined_scores[:k], combined_scores[-k:]
    top_k_doc_ids, bottom_k_doc_ids = [x[0] for x in top_k_docs], [
        x[0] for x in bottom_k_docs
    ]
    return top_k_doc_ids, bottom_k_doc_ids


def get_top_k_documents_by_cosine(new_q_data, new_d_data, k=6, batch_size=4096):
    device_count = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(device_count)]
    print(f"Using GPUs: {devices}")

    results = {}
    doc_list = list(new_d_data.values())
    for qcnt, query_id in enumerate(new_q_data.keys()):
        query_emb = query = new_q_data[query_id]["EMB"]
        top_k_doc_ids, bottom_k_doc_ids = get_top_k_documents_cosine(
            query_emb, doc_list, k, devices, batch_size
        )
        results[query_id] = {
            "top1": top_k_doc_ids[:1],
            "bottom6": bottom_k_doc_ids[-6:],
        }
        if qcnt % 10 == 0:
            print(f"#{qcnt} retrieving is done.")
    return results


def make_query_cos_samples(model, queries, docs, sampling_size_per_query=100):
    print("make_query_cos_samples started.")
    doc_list = list(docs.values())
    corpus = [doc["text"] for doc in doc_list]
    bm25 = BM25Okapi(corpus=corpus, tokenizer=preprocess)
    doc_ids = [doc["doc_id"] for doc in doc_list]
    candidate_doc_ids = set()
    for i, query in enumerate(queries):
        print(f"{i}th query * {sampling_size_per_query}")
        query_tokens = preprocess(query["query"])
        scores = bm25.get_scores(query_tokens)
        sorted_indices = np.argsort(scores)[::-1]
        top_k_indices, bottom_k_indices = (
            sorted_indices[:sampling_size_per_query],
            sorted_indices[-sampling_size_per_query:],
        )
        top_k_doc_ids, bottom_k_doc_ids = [doc_ids[_id] for _id in top_k_indices], [
            doc_ids[_id] for _id in bottom_k_indices
        ]
        candidate_doc_ids.update(top_k_doc_ids)
        candidate_doc_ids.update(bottom_k_doc_ids)
    doc_list = [docs[doc_id] for doc_id in candidate_doc_ids]

    query_embs, doc_embs = encode_data_mean_pooling(model, queries, doc_list)
    result = get_top_k_documents_by_cosine(query_embs, doc_embs, 6)
    print("make_query_cos_samples finished.")
    return result


# def get_top_k_documents_gpu(new_q_data, docs, query_id, k, devices, batch_size=4096):
#     query_data_item = new_q_data[query_id]
#     query_token_embs = query_data_item["TOKEN_EMBS"]

#     docs_cnt = len(docs)
#     num_gpus = len(devices)
#     batch_indices = [
#         list(range(i * batch_size, min((i + 1) * batch_size, docs_cnt)))
#         for i in range((docs_cnt + batch_size - 1) // batch_size)
#     ]
#     gpu_batch_indices = [batch_indices[i::num_gpus] for i in range(num_gpus)]

#     def process(query_token_embs, gpu_batch_indices, device):
#         query_token_embs = query_token_embs.clone().unsqueeze(0).to(device)
#         regl_scores = []
#         for batch in gpu_batch_indices:
#             start_idx, end_idx = batch[0], batch[-1] + 1
#             print(f"{device}| Processing batch {start_idx}-{end_idx}")
#             combined_embs = torch.stack(
#                 [doc["TOKEN_EMBS"].to(device) for doc in docs[start_idx:end_idx]], dim=0
#             )
#             # print(f'query_token_embs:{query_token_embs.shape}, combined_embs:{combined_embs.shape}')
#             regl_score = calculate_S_qd_regl_batch(
#                 query_token_embs, combined_embs, device
#             )
#             regl_scores.extend(
#                 [
#                     (doc["doc_id"], regl_score[idx].item())
#                     for idx, doc in enumerate(docs[start_idx:end_idx])
#                 ]
#             )
#         return regl_scores

#     with ThreadPoolExecutor(max_workers=num_gpus) as executor:
#         args = [
#             (query_token_embs, gpu_batch_indices[i], devices[i])
#             for i in range(num_gpus)
#         ]
#         results = list(executor.map(lambda p: process(*p), args))

#     combined_regl_scores = [score for gpu_scores in results for score in gpu_scores]
#     combined_regl_scores = sorted(
#         combined_regl_scores, key=lambda x: x[1], reverse=True
#     )
#     top_k_regl_docs, bottom_k_regl_docs = (
#         combined_regl_scores[:k],
#         combined_regl_scores[-k:],
#     )
#     top_k_regl_doc_ids, bottom_k_regl_doc_ids = [x[0] for x in top_k_regl_docs], [
#         x[0] for x in bottom_k_regl_docs
#     ]
#     return top_k_regl_doc_ids, bottom_k_regl_doc_ids


def get_top_k_documents_gpu_batch(
    new_q_data_batch, docs, query_ids, k, devices, batch_size=2048
):
    queries_token_embs = [new_q_data_batch[qid]["TOKEN_EMBS"] for qid in query_ids]
    queries_token_embs = torch.stack(
        queries_token_embs
    )  # (qbatch_size, seqlen, hidden)
    docs_cnt = len(docs)
    num_gpus = len(devices)
    batch_indices = [
        list(range(i * batch_size, min((i + 1) * batch_size, docs_cnt)))
        for i in range((docs_cnt + batch_size - 1) // batch_size)
    ]
    gpu_batch_indices = [batch_indices[i::num_gpus] for i in range(num_gpus)]

    def process(queries_token_embs, gpu_batch_indices, device):
        queries_token_embs = queries_token_embs.clone().to(
            device
        )  # (qbatch_size, seqlen, hidden)
        all_regl_scores = [[] for _ in range(queries_token_embs.size(0))]  # 쿼리마다 별도로 저장
        for batch in gpu_batch_indices:
            start_idx, end_idx = batch[0], batch[-1] + 1
            print(f"{device}| Processing batch {start_idx}-{end_idx}")
            combined_embs = torch.stack(
                [doc["TOKEN_EMBS"].to(device) for doc in docs[start_idx:end_idx]], dim=0
            )  # (batch_size, seqlen, hidden)
            regl_score = calculate_S_qd_regl_batch_batch(
                queries_token_embs, combined_embs, device
            )  # (qbatch_size, batch_size)
            for q_idx in range(queries_token_embs.size(0)):
                all_regl_scores[q_idx].extend(
                    [
                        (
                            docs[start_idx + d_idx]["doc_id"],
                            regl_score[q_idx, d_idx].item(),
                        )
                        for d_idx in range(end_idx - start_idx)
                    ]
                )
        return all_regl_scores

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        args = [
            (queries_token_embs, gpu_batch_indices[i], devices[i])
            for i in range(num_gpus)
        ]
        results = list(executor.map(lambda p: process(*p), args))
    # 각 쿼리별로 결과 합치기
    qbatch_size = len(query_ids)
    combined_regl_scores_per_query = [[] for _ in range(qbatch_size)]
    for gpu_result in results:
        for q_idx, scores in enumerate(gpu_result):
            combined_regl_scores_per_query[q_idx].extend(scores)
    topk_bottomk_results = {}
    for q_idx, query_id in enumerate(query_ids):
        combined_regl_scores = combined_regl_scores_per_query[q_idx]
        combined_regl_scores = sorted(
            combined_regl_scores, key=lambda x: x[1], reverse=True
        )
        top_k_regl_docs = combined_regl_scores[:1]
        bottom_k_regl_docs = combined_regl_scores[-k:]
        top_k_regl_doc_ids = [x[0] for x in top_k_regl_docs]
        bottom_k_regl_doc_ids = [x[0] for x in bottom_k_regl_docs]
        topk_bottomk_results[query_id] = {
            "top1": top_k_regl_doc_ids,
            "bottom6": bottom_k_regl_doc_ids,
        }
    return topk_bottomk_results


# def get_top_k_documents(new_q_data, new_d_data, k=6, batch_size=4096):
#     device_count = torch.cuda.device_count()
#     devices = [torch.device(f"cuda:{i}") for i in range(device_count)]
#     print(f"Using GPUs: {devices}")


#     docs = list(new_d_data.values())
#     results = {}
#     for qcnt, query_id in enumerate(new_q_data.keys()):
#         ttop_k_doc_ids, bottom_k_doc_ids = get_top_k_documents_gpu(
#             new_q_data, docs, query_id, k, devices, batch_size
#         )
#         results[query_id] = {
#             "top1": ttop_k_doc_ids[:1],
#             "bottom6": bottom_k_doc_ids[-6:],
#         }
#         if qcnt % 10 == 0:
#             print(f"#{qcnt} retrieving is done.")
#     return results
def get_top_k_documents(new_q_data, new_d_data, k=10, batch_size=2048, qbatch_size=10):
    device_count = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(device_count)]
    print(f"ablation.incremental get_top_k_documents | Using GPUs: {devices}")
    docs = list(new_d_data.values())
    query_ids = list(new_q_data.keys())
    results = {}
    for i in range(0, len(query_ids), qbatch_size):
        batch_query_ids = query_ids[i : i + qbatch_size]
        batch_q_data = {qid: new_q_data[qid] for qid in batch_query_ids}
        batch_results = get_top_k_documents_gpu_batch(
            batch_q_data, docs, batch_query_ids, k, devices, batch_size
        )
        results.update(batch_results)
        print(f"# {i} ~ {i+len(batch_query_ids)-1} retrieving is done.")
    return results


def make_query_term_samples(model, queries, docs, sampling_size_per_query=50):
    print(f"make_query_term_samples started.({sampling_size_per_query})")
    doc_list = list(docs.values())
    # corpus = [doc["text"] for doc in doc_list]
    # bm25 = BM25Okapi(corpus=corpus, tokenizer=preprocess)
    # doc_ids = [doc["doc_id"] for doc in doc_list]
    # candidate_doc_ids = set()
    # for i, query in enumerate(queries):
    #     print(f"{i}th query * {sampling_size_per_query}")
    #     query_tokens = preprocess(query["query"])
    #     scores = bm25.get_scores(query_tokens)
    #     sorted_indices = np.argsort(scores)[::-1]
    #     top_k_indices = sorted_indices[:sampling_size_per_query]
    #     top_k_doc_ids = [doc_ids[_id] for _id in top_k_indices]
    #     candidate_doc_ids.update(top_k_doc_ids)
    # doc_list = [docs[doc_id] for doc_id in candidate_doc_ids]

    query_embs, doc_embs = renew_data(queries, doc_list)
    result = get_top_k_documents(query_embs, doc_embs, 6)
    print("make_query_term_samples finished.")
    return result
