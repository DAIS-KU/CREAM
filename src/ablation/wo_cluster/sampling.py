import numpy as np

from clusters import renew_data
from data import BM25Okapi
from functions import (
    calculate_S_qd_regl_batch,
    get_passage_embeddings,
    get_top_k_documents,
)


def preprocess(corpus, max_length=256):
    # corpus = corpus.lower()
    # corpus = "".join([char for char in corpus if char not in string.punctuation])
    # tokens = word_tokenize(corpus)
    # max_len = min(max_length, len(tokens))
    # tokens = tokens[:max_len]
    tokens = corpus.split(" ")
    return tokens


def make_query_psuedo_answers_wo_cluster(
    model_path, queries, doc_list, k=1, sampling_size_per_query=100
):
    print("make_query_psuedo_answers_wo_cluster started.")
    # new_q_data, new_d_data = renew_data(
    #     queries=queries,
    #     documents=doc_list,
    #     model_path=model_path,
    #     nbits=12,
    #     renew_q=True,
    #     renew_d=True,
    #     use_tensor_key=True,
    # )
    # result = get_top_k_documents(new_q_data, new_d_data, k)
    # result = {key: value[0] for key, value in result.items()}
    doc_list = list(docs.values())
    corpus = [doc["text"] for doc in doc_list]
    bm25 = BM25Okapi(corpus=corpus, tokenizer=preprocess)
    doc_ids = [doc["doc_id"] for doc in doc_list]
    result = {}
    for i, query in enumerate(queries):
        print(f"{i}th query * {sampling_size_per_query}")
        query_tokens = preprocess(query["query"])
        scores = bm25.get_scores(query_tokens)
        top_k_indices = np.argsort(scores)[::-1][:sampling_size_per_query]

        query_token_emb = get_passage_embeddings(
            model, query["query"], model.device, max_length=256
        )
        doc_batches = [docs[doc_ids[i]]["text"] for i in top_k_indices]
        doc_embs = get_passage_embeddings(
            model, doc_batches, model.device, max_length=256
        )
        regl_scores = calculate_S_qd_regl_batch(
            query_token_emb, doc_embs, query_token_emb.device
        )
        # TODO 검토 필요
        regl_scores = list(zip([doc_ids[i] for i in top_k_indices], regl_scores))
        regl_scores = sorted(regl_scores, key=lambda x: x[1], reverse=True)
        top_regl_doc_id = regl_scores[0][0]
        result[query["qid"]] = top_regl_doc_id
    print("make_query_psuedo_answers_wo_cluster finished.")
    return result
