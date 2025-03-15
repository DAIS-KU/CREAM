import random

import numpy as np

from .loader import read_jsonl, read_jsonl_as_dict


class Stream:
    def __init__(
        self,
        session_number,
        query_path,
        doc_path,
        prev_docs,
        warmingup_rate,
        sampling_rate=0.5,
        stream_batch_size=512,
    ):
        queries = read_jsonl(query_path, True)
        docs = read_jsonl_as_dict(doc_path, id_field="doc_id")
        doc_list = list(docs.values())

        random.shuffle(queries)
        random.shuffle(doc_list)
        doc_sampled_cnt = int(len(doc_list) * sampling_rate)
        doc_list = doc_list[:doc_sampled_cnt]

        self.queries = queries
        self.docs = {doc["doc_id"]: doc for doc in doc_list}
        self.docs = {**docs, **prev_docs} if prev_docs else docs

        if session_number == 0:
            warmingup_cnt = int(len(doc_list) * warmingup_rate)
            self.initial_docs = doc_list[:warmingup_cnt]
            doc_list = doc_list[warmingup_cnt:]
        else:
            self.initial_docs = []

        self.stream_docs = [
            doc_list[i : i + stream_batch_size]
            for i in range(0, len(doc_list), stream_batch_size)
        ]
        self.index = 0
        print(
            f"queries #{len(self.queries)}, initial_docs #{len(self.initial_docs)}, #stream {len(self.stream_docs)}, {min(map(len, self.stream_docs))}-{max(map(len, self.stream_docs))}"
        )

    def get_stream(self, index):
        return self.queries[index], self.stream_docs[index]

    def get_stream_size(self):
        return len(self.stream_docs)
