import json
from bm25 import BM25Okapi, preprocess
import random


def load_tsv(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
        headers = lines[0].split("\t")
        return [dict(zip(headers, line.split("\t"))) for line in lines[1:]]


def load_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def read_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def save_tsv(filepath, headers, rows):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(str(row[h]) for h in headers) + "\n")


def convert_jsonl_to_tsv(query_file, doc_file, queries_tsv, documents_tsv):
    # 문서 처리
    doc_data = read_jsonl(doc_file)
    doc_id_map = {}
    doc_rows = []
    for idx, doc in enumerate(doc_data, start=1):
        doc_id_map[doc["doc_id"]] = idx
        doc_rows.append({"id": idx, "text": doc["text"]})

    # 쿼리 처리
    query_data = read_jsonl(query_file)
    query_rows = []
    for idx, query in enumerate(query_data, start=1):
        mapped_pids = [
            str(doc_id_map[pid])
            for pid in query.get("answer_pids", [])
            if pid in doc_id_map
        ]
        query_rows.append(
            {"id": idx, "query": query["query"], "answer_pids": ",".join(mapped_pids)}
        )

    # TSV 저장
    save_tsv(queries_tsv, ["id", "query", "answer_pids"], query_rows)
    save_tsv(documents_tsv, ["id", "text"], doc_rows)


def build_triples_bm25(
    queries_tsv, documents_tsv, queries_jsonl, output_file, num_triples=6, top_k=50
):
    queries = load_tsv(queries_tsv)
    documents = load_tsv(documents_tsv)
    queries_json = load_jsonl(queries_jsonl)

    # 문서 ID → 텍스트 매핑
    doc_id_to_text = {int(d["id"]): d["text"] for d in documents}
    doc_ids = list(doc_id_to_text.keys())
    corpus = [preprocess(doc_id_to_text[doc_id]) for doc_id in doc_ids]

    # BM25 모델 생성
    bm25 = BM25Okapi(corpus)

    triples = []

    for qrow, qjson in zip(queries, queries_json):
        qid = int(qrow["id"])
        query_text = qrow["query"]
        pos_ids = (
            list(map(int, qrow["answer_pids"].split(",")))
            if qrow["answer_pids"]
            else []
        )
        if not pos_ids:
            continue  # skip if no positive

        # BM25 ranking
        tokenized_query = preprocess(query_text)
        scores = bm25.get_scores(tokenized_query)
        ranked_doc_ids = [
            doc_ids[i]
            for i in sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
        ]
        neg_candidates = [pid for pid in ranked_doc_ids if pid not in pos_ids][:top_k]

        for _ in range(num_triples):
            pos = random.choice(pos_ids)
            if not neg_candidates:
                continue
            neg = random.choice(neg_candidates)
            triples.append(f"{qid}\t{pos}\t{neg}")
    # TSV 저장
    with open(output_file, "w", encoding="utf-8") as f:
        for line in triples:
            f.write(line + "\n")

    print(f"Saved {len(triples)} triples to {output_file}")


if __name__ == "__main__":
    # convert_jsonl_to_tsv(
    #     query_file='/home/work/retrieval/data/datasetL/test_session0_queries.jsonl',
    #     doc_file='/home/work/retrieval/data/datasetL/test_session0_docs.jsonl',
    #     queries_tsv='/home/work/retrieval/data/datasetL/colbert/queries.tsv',
    #     documents_tsv='/home/work/retrieval/data/datasetL/colbert/collection.tsv'
    # )
    build_triples_bm25(
        queries_tsv="/home/work/retrieval/data/datasetL/colbert/queries.tsv",
        documents_tsv="/home/work/retrieval/data/datasetL/colbert/collection.tsv",
        queries_jsonl="/home/work/retrieval/data/datasetL/test_session0_queries.jsonl",
        output_file="/home/work/retrieval/data/datasetL/colbert/triples.tsv",
        num_triples=6,
        top_k=50,
    )
