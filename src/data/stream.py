import random
import string

import numpy as np
from nltk.tokenize import word_tokenize

from .bm25 import BM25Okapi
from .loader import read_jsonl, read_jsonl_as_dict


def preprocess(corpus, max_length=256):
    # corpus = corpus.lower()
    # corpus = "".join([char for char in corpus if char not in string.punctuation])
    # tokens = word_tokenize(corpus)
    # max_len = min(max_length, len(tokens))
    # tokens = tokens[:max_len]
    tokens = corpus.split(" ")
    return tokens


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
    ):
        queries = read_jsonl(query_path, True)
        random.shuffle(queries)
        self.queries = queries
        query_docs = {
            query["qid"]: {
                "doc_id": query["qid"],
                "text": query["query"],
                "is_query": True,
            }
            for query in queries
        }
        query_list = list(query_docs.values())

        docs = read_jsonl_as_dict(doc_path, id_field="doc_id")
        doc_list = self.filter(queries, docs, sampling_rate, sampling_size_per_query)
        self.docs = {
            doc["doc_id"]: {
                "doc_id": doc["doc_id"],
                "text": doc["text"],
                "is_query": False,
            }
            for doc in doc_list
        }
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

    def filter(self, queries, docs, sampling_rate=None, sampling_size_per_query=None):
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
            corpus = [doc["text"] for doc in doc_list]
            bm25 = BM25Okapi(corpus=corpus, tokenizer=preprocess)
            doc_ids = [doc["doc_id"] for doc in doc_list]
            candidate_ids = set()
            for i, query in enumerate(queries):
                print(f"{i}th query * {sampling_size_per_query}")
                query_tokens = preprocess(query["query"])
                scores = bm25.get_scores(query_tokens)
                top_k_indices = np.argsort(scores)[::-1][:sampling_size_per_query]
                candidate_ids.update([doc_ids[i] for i in top_k_indices])
            doc_list = [docs[doc_id] for doc_id in candidate_ids]
            return doc_list
        else:
            print(f"Documents are raw.")
            doc_list = list(docs.values())
            return doc_list
