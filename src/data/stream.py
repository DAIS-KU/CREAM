import random

import numpy as np

from .loader import read_jsonl, read_jsonl_as_dict


class Stream:
    def __init__(self, session_number, query_path, doc_path, warmingup_rate, prev_docs):
        queries = read_jsonl(query_path, True)[:8]
        docs = read_jsonl_as_dict(doc_path, id_field="doc_id")
        doc_list = list(docs.values())[:2400]

        random.shuffle(queries)
        random.shuffle(doc_list)
        self.queries = queries
        self.docs = {**docs, **prev_docs} if prev_docs else docs

        if session_number == 0:
            warmingup_cnt = int(len(doc_list) * warmingup_rate)
            self.initial_docs = doc_list[:warmingup_cnt]
            self.stream_docs = np.array_split(doc_list[warmingup_cnt:], len(queries))
        else:
            self.initial_docs = []
            self.stream_docs = np.array_split(doc_list, len(queries))

        self.index = 0
        print(
            f"queries #{len(self.queries)}, initial_docs #{len(self.initial_docs)}, stream_docs #{len(self.stream_docs)}, {min(map(len, self.stream_docs))}-{max(map(len, self.stream_docs))}"
        )

    def get_stream(self, index):
        return self.queries[index], self.stream_docs[index]

    def get_stream_size(self):
        return len(self.stream_docs)
