import random
import string

import numpy as np
from .encode import renew_data
import time
from data import BM25Okapi, read_jsonl, read_jsonl_as_dict, preprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def collect_candidates(
    bm25: BM25Okapi, doc_ids, query, sampling_size_per_query, include_answer
):
    query_tokens = preprocess(query["query"])
    scores = bm25.get_scores(query_tokens)
    # 상위 k개 인덱스 추출
    top_k_idx = np.argpartition(scores, -sampling_size_per_query)[
        -sampling_size_per_query:
    ]
    top_k_indices = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
    cand = {doc_ids[i] for i in top_k_indices}
    if include_answer:
        cand.update(query["answer_pids"])
    return cand


def filter_docs_parallel(
    docs,
    doc_list,
    queries,
    sampling_size_per_query,
    include_answer=False,
    max_workers=8,
):
    corpus = [doc["text"] for doc in docs.values()]
    bm25 = BM25Okapi(corpus=corpus, tokenizer=preprocess)
    doc_ids = [doc["doc_id"] for doc in doc_list]
    candidate_ids = set()

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                collect_candidates,
                bm25,
                doc_ids,
                query,
                sampling_size_per_query,
                include_answer,
            )
            for query in queries
        ]
        for fut in as_completed(futures):
            candidate_ids.update(fut.result())
    end_time = time.time()
    print(f"Filtering (parallel): {end_time - start_time:.2f} sec.")
    return [docs[doc_id] for doc_id in candidate_ids]


class Stream:
    def __init__(
        self,
        session_number,
        query_path,
        doc_path,
        warming_up_method,
        prev_docs=None,
        warmingup_rate=None,
        sampling_rate=None,
        sampling_size_per_query=None,
        query_stream_batch_size=256,
        doc_stream_batch_size=1024,
        include_answer=False,
    ):
        # Read raw data
        queries = read_jsonl(query_path, True)
        docs = read_jsonl_as_dict(doc_path, id_field="doc_id")
        print(f"queries:{len(queries)}, docs:{len(docs)}")

        # Filter document(doc_id, text)
        # documents = self.filter(
        #     queries, docs, sampling_rate, sampling_size_per_query, include_answer
        # )
        documents = self.read_filtered_docs(session_number)

        print(f"queries:{len(queries)}, documents:{len(documents)}")
        # Encode (doc_id, text, token_embs, is_query)
        query_docs, doc_docs = renew_data(queries=queries, documents=documents)
        # Export list to make streams
        query_list = list(query_docs.values())
        doc_list = list(doc_docs.values())
        print(f"query_list:{len(query_list)}, doc_list:{len(doc_list)}")

        # Prepare training (if not N)
        # random.shuffle(query_list)
        self.queries = query_list
        self.docs = doc_docs
        self.docs.update(query_docs)
        if prev_docs is not None:
            self.docs.update(prev_docs)

        if warming_up_method == "initial_cluster":
            if session_number == 0:
                if warmingup_rate is not None:
                    print(f"Documents are sampled for warming up.")
                    warmingup_cnt = int(len(doc_list) * warmingup_rate)
                    self.initial_docs = doc_list[:warmingup_cnt]
                    doc_list = query_list + doc_list[warmingup_cnt:]
                    self.stream_queries = []
                else:
                    raise ValueError(
                        "Invalid initial_cluster condition and parameters."
                    )
            else:
                self.initial_docs = []
                self.stream_queries = []
                doc_list = query_list + doc_list
        elif warming_up_method == "query_seed":
            self.initial_docs = []
            self.stream_queries = [query_list[:query_stream_batch_size]]
            doc_list = query_list[query_stream_batch_size:] + doc_list
        elif warming_up_method == "stream_seed" or warming_up_method == "none":
            self.initial_docs = []
            self.stream_queries = []
            doc_list = query_list + doc_list
        elif warming_up_method == "eval":
            self.initial_docs = []
            self.stream_queries = []
            doc_list = []
        else:
            raise NotImplementedError(
                f"Unsupported warming_up_method: {warming_up_method}"
            )
        random.shuffle(doc_list)
        self.stream_docs = [
            doc_list[i : i + doc_stream_batch_size]
            for i in range(0, len(doc_list), doc_stream_batch_size)
        ]

        print(
            f"queries #{len(self.queries)}, documents #{len(self.docs)} initial_docs #{len(self.initial_docs)}"
        )
        if len(self.stream_queries) > 0:
            print(
                f"#stream_queries {len(self.stream_queries)}, stream_queries size {min(map(len, self.stream_queries))}-{max(map(len, self.stream_queries))}"
            )
        if len(self.stream_docs) > 0:
            print(
                f"#doc_stream {len(self.stream_docs)}, stream_docs size {min(map(len, self.stream_docs))}-{max(map(len, self.stream_docs))}"
            )

    def read_filtered_docs(self, session_number):
        # filtered_doc_path = f"/home/work/.default/huijeong/cream/data/datasetL_large_share/train_session{session_number}_docs_filtered.jsonl"
        filtered_doc_path = f"/home/work/.default/huijeong/cream/data/datasetM_large_share/train_session{session_number}_docs_filtered.jsonl"
        # filtered_doc_path = f"/home/work/.default/huijeong/cream/data/datasetN_large/train_session{session_number}_docs_filtered.jsonl"
        print(f"Read from: {filtered_doc_path}")
        filtered_doc_list = read_jsonl(filtered_doc_path, is_query=False)
        return filtered_doc_list

    def filter(
        self,
        queries,
        docs,
        sampling_rate=None,
        sampling_size_per_query=None,
        include_answer=False,
    ):
        if sampling_rate is not None:
            print(f"Documents are sampled for down-scaling.")
            doc_list = list(docs.values())
            random.shuffle(doc_list)
            doc_sampled_cnt = int(len(doc_list) * sampling_rate)
            doc_list = doc_list[:doc_sampled_cnt]
            return doc_list
        elif sampling_size_per_query is not None:
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
                candidates = [
                    doc_ids[i] for i in top_k_indices
                ]  # [:sampling_size_per_query]
                candidate_ids.update(candidates)
                if include_answer:
                    print(f"Add {i}th query answsers {len(query['answer_pids'])}")
                    candidate_ids.update(query["answer_pids"])
                # self.query_result[query["qid"]]=candidates
            doc_list = [docs[doc_id] for doc_id in candidate_ids]
            end_time = time.time()
            print(f"Filtering : {end_time-start_time} sec.")
            return doc_list
        else:
            print(f"Documents are raw.")
            doc_list = list(docs.values())
            return doc_list
