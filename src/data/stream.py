import random

from .loader import read_jsonl


class Stream:
    def __init__(self, session_number, query_path, doc_path, warmingup_rate):
        queries = read_jsonl(query_path, True)
        docs = read_jsonl_as_dict(doc_path, id_field="doc_id")

        random.shuffle(queries)
        random.shuffle(docs)
        self.queries = queries
        self.docs = docs

        if session_number == 0:
            warmingup_cnt = len(docs) * warmingup_rate
            self.initial_docs = docs[:warmingup_cnt]
            docs = docs[warmingup_cnt:]

        batch_size = len(docs) // len(queries)
        self.stream_docs = [
            docs[i : i + batch_size] for i in range(0, len(docs), batch_size)
        ]
        self.index = 0

    def get_stream(self, index):
        return self.queries[index], self.stream_docs[index]

    def get_stream_size(self):
        return len(self.stream_docs)
