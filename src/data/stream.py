import random

import numpy as np

from .loader import read_jsonl, read_jsonl_as_dict


class Stream:
    def __init__(self, session_number, query_path, doc_path, warmingup_rate):
        queries = read_jsonl(query_path, True)
        docs = read_jsonl_as_dict(doc_path, id_field="doc_id")
        doc_list = list(docs.values())

        random.shuffle(queries)
        random.shuffle(doc_list)
        self.queries = queries
        self.docs = docs

        if session_number == 0:
            warmingup_cnt = int(len(docs) * warmingup_rate)
            self.initial_docs = doc_list[:warmingup_cnt]
            stream_docs = doc_list[warmingup_cnt:]

        batch_size = len(stream_docs) // len(queries)
        self.stream_docs = np.array_split(stream_docs, len(queries))
        self.index = 0
        print(
            f"queries #{len(self.queries)}, initial_docs #{len(self.initial_docs)}, stream_docs #{len(self.stream_docs)}"
        )

    def get_stream(self, index):
        return self.queries[index], self.stream_docs[index]

    def get_stream_size(self):
        return len(self.stream_docs)
