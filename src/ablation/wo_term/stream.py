import random
import string

import torch
import numpy as np
from functions import renew_data_mean_pooling

from data import BM25Okapi, read_jsonl, read_jsonl_as_dict, preprocess
from transformers import BertModel, BertTokenizer

num_gpus = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]


def model_builder(model_path):
    model = BertModel.from_pretrained(
        "/home/work/.default/huijeong/bert_local"
    ).to(devices[-1])
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=devices[-1]))
    model.eval()
    return model


class Stream:
    def __init__(
        self,
        session_number,
        model_path,
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
        # Read raw data
        queries = read_jsonl(query_path, True)
        docs = read_jsonl_as_dict(doc_path, id_field="doc_id")
        print(f"queries:{len(queries)}, docs:{len(docs)}")

        # Filter document(doc_id, text)
        # documents = self.filter(queries, docs, sampling_rate, sampling_size_per_query)
        documents = self.read_filtered_docs(session_number)
        print(f"queries:{len(queries)}, documents:{len(documents)}")
        # Encode (doc_id, text, token_embs, is_query)
        query_docs, doc_docs = renew_data_mean_pooling(
            model_builder=model_builder,
            model_path=model_path,
            queries_data=queries,
            documents_data=documents,
        )
        # Export list to make streams
        query_list = list(query_docs.values())
        doc_list = list(doc_docs.values())
        print(f"query_list:{len(query_list)}, doc_list:{len(doc_list)}")

        # Prepare training
        random.shuffle(query_list)
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

    def read_filtered_docs(self, session_number):
        filtered_doc_path = f"/home/work/.default/huijeong/data/msmarco_session/train_session{session_number}_docs.jsonl"
        filtered_doc_list = read_jsonl(filtered_doc_path, is_query=False)
        return filtered_doc_list
