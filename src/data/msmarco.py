import csv
import json
from typing import Union
from datasets import load_dataset, concatenate_datasets


def generate_msmarco_subset_files(
    domain,
    dataset,
    tsv_path: str,
    output_docs_path: str = "docs.jsonl",
    output_queries_path: str = "queries.jsonl",
    max_text_length: int = 4096,
):
    qid2query = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                qid, query = int(row[0]), row[1]
                qid2query[qid] = query
    print("Indexing dataset by query_id...")
    qid2example = {ex["query_id"]: ex for ex in dataset if ex["query_id"] in qid2query}
    print(f"qid2example: {len(qid2example)}")

    doc_entries = {}
    query_entries = {}

    for qid, query_text in qid2query.items():
        example = qid2example.get(qid)
        query_id = f"{domain}-{qid}"
        if not example:
            print(f"Cannot find doc about {qid}")
            continue

        positives, negatives = [], []
        for idx, (is_sel, passage) in enumerate(
            zip(example["passages"]["is_selected"], example["passages"]["passage_text"])
        ):
            doc_id = f"{domain}-{qid}_{idx}"
            if doc_id not in doc_entries:
                doc_entries[doc_id] = {
                    "doc_id": doc_id,
                    "text": passage[:max_text_length],
                }
            if is_sel == 1:
                positives.append(doc_id)
            else:
                negatives.append(doc_id)

        if positives:
            query_entries[query_id] = {
                "qid": query_id,
                "query": query_text,
                "answer_pids": positives,
                "negatives": negatives,
            }

    # 5. JSONL로 저장
    with open(output_docs_path, "w", encoding="utf-8") as f_doc:
        for doc in doc_entries.values():
            f_doc.write(json.dumps(doc, ensure_ascii=False) + "\n")

    with open(output_queries_path, "w", encoding="utf-8") as f_query:
        for query in query_entries.values():
            f_query.write(json.dumps(query, ensure_ascii=False) + "\n")

    print(f"Saved {len(query_entries)} queries to {output_queries_path}")
    print(f"Saved {len(doc_entries)} documents to {output_docs_path}")


if __name__ == "__main__":
    print("Loading MSMARCO v2.1 splits...")
    train = load_dataset("microsoft/ms_marco", "v2.1", split="train")
    val = load_dataset("microsoft/ms_marco", "v2.1", split="validation")
    test = load_dataset("microsoft/ms_marco", "v2.1", split="test")
    dataset = concatenate_datasets([train, val, test])
    print(f"Dataset loaded. Total examples: {len(dataset)}")
    # print(f"ex: {dataset[0]}")

    for i in range(5):
        generate_msmarco_subset_files(
            domain=f"domain{i}",
            dataset=dataset,
            tsv_path=f"/home/work/retrieval/data/raw/msmarco/queries_{i}.tsv",
            output_docs_path=f"/home/work/retrieval/data/raw/msmarco/domain{i}_docs.jsonl",
            output_queries_path=f"/home/work/retrieval/data/raw/msmarco/domain{i}_queries.jsonl",
        )
