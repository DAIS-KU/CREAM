from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import torch
from transformers import BertTokenizer

from data import BM25Okapi
from functions import process_batch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


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


def make_query_cos_samples(model, queries, docs, sampling_size_per_query=100):
    print("make_query_cos_samples started.")
    doc_list = list(docs.values())[:1000]
    corpus = [doc["text"] for doc in doc_list]
    bm25 = BM25Okapi(corpus=corpus, tokenizer=preprocess)
    doc_ids = [doc["doc_id"] for doc in doc_list]

    result = {}
    query_embs, doc_embs = encode_data_mean_pooling(model, queries, doc_list)

    for i, query in enumerate(queries):
        print(f"{i}th query * {sampling_size_per_query}")
        query_tokens = preprocess(query["query"])
        scores = bm25.get_scores(query_tokens)
        sorted_indices = np.argsort(scores)[::-1]
        top_k_indices = sorted_indices[:sampling_size_per_query]
        bottom_k_indices = sorted_indices[sampling_size_per_query:]

        query_emb = query_embs[query["qid"]]["EMB"]
        top_k_doc_embs = torch.stack(
            [doc_embs[doc_ids[_id]]["EMB"] for _id in top_k_indices]
        )
        bottom_k_doc_embs = torch.stack(
            [doc_embs[doc_ids[_id]]["EMB"] for _id in bottom_k_indices]
        )
        top_k_similarities = torch.nn.functional.cosine_similarity(
            top_k_doc_embs, query_emb.unsqueeze(0), dim=1
        )
        bottom_k_similarities = torch.nn.functional.cosine_similarity(
            bottom_k_doc_embs, query_emb.unsqueeze(0), dim=1
        )
        top1_index = top_k_indices[torch.argmax(top_k_similarities)].item()
        top1_docid = doc_ids[top1_index]
        bottom6_indices = bottom_k_indices[
            torch.argsort(bottom_k_similarities)[:6]
        ].tolist()
        bottom6_docids = [doc_ids[idx] for idx in bottom6_indices]
        result[query["qid"]] = {"top1": top1_docid, "bottom6": bottom6_docids}

    print("make_query_cos_samples finished.")
    return result
