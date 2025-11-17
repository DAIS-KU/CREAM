from loader import read_jsonl, read_jsonl_as_dict
import json
from bm25 import BM25Okapi, preprocess
from collections import defaultdict
import time
import numpy as np


def save_dicts_to_jsonl(dict_list, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in dict_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")


def prefilter(queries, docs, sampling_size_per_query):
    print(f"Documents are passing through BM25.")
    doc_list = list(docs.values())
    # return filter_docs_parallel(docs, doc_list, queries, sampling_size_per_query)
    corpus = [doc["text"] for doc in doc_list]
    bm25 = BM25Okapi(corpus=corpus, tokenizer=preprocess)
    doc_ids = [doc["doc_id"] for doc in doc_list]
    candidate_ids = set()
    start_time = time.time()
    for i, query in enumerate(queries):
        print(f"{i}th query * {sampling_size_per_query}")
        query_tokens = preprocess(query["query"])
        scores = bm25.get_scores(query_tokens)
        # 상위 k개만 찾아서 나중에 정렬
        top_k_idx = np.argpartition(scores, -sampling_size_per_query)[
            -sampling_size_per_query:
        ]
        top_k_indices = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        # top_k_indices = np.argsort(scores)[::-1]
        candidates = [doc_ids[i] for i in top_k_indices]  # [:sampling_size_per_query]
        candidate_ids.update(candidates)
        # if include_answer:
        #     print(f"Add {i}th query answsers {len(query['answer_pids'])}")
        #     candidate_ids.update(query["answer_pids"])
        # self.query_result[query["qid"]]=candidates
    doc_list = [docs[doc_id] for doc_id in candidate_ids]
    end_time = time.time()
    print(f"Filtering : {end_time-start_time} sec.")
    return doc_list


def make_prestream_docs(
    start_session_number=0, end_sesison_number=10, sampling_size_per_query=30
):
    for session_number in range(start_session_number, end_sesison_number):
        query_path = f"/home/work/.default/huijeong/cream/data/datasetM_large_share/train_session{session_number}_queries.jsonl"
        doc_path = f"/home/work/.default/huijeong/cream/data/datasetM_large_share/train_session{session_number}_docs.jsonl"
        queries = read_jsonl(query_path, True)
        docs = read_jsonl_as_dict(doc_path, id_field="doc_id")
        print(f"queries:{len(queries)}, docs:{len(docs)}")

        filtered_doc_list = prefilter(queries, docs, sampling_size_per_query)
        filtered_doc_path = f"/home/work/.default/huijeong/cream/data/datasetM_large_share/train_session{session_number}_docs_filtered_30.jsonl"
        save_dicts_to_jsonl(filtered_doc_list, filtered_doc_path)


if __name__ == "__main__":
    make_prestream_docs()
