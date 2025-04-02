import csv
import json

from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, Dataset


def generate_docs():
    all_data = load_dataset("microsoft/ms_marco", "v2.1")
    all_data = concatenate_datasets([all_data[split] for split in all_data])
    all_docs = {
        data["query_id"]: {"doc_id": f'd{data["query_id"]}', "text": data["answers"]}
        for data in list(all_data)
    }
    print(f"all_docs: {len(all_docs.keys())}")

    for num in range(5):
        query_file_path = f"/home/work/retrieval/data/msmarco/{num}_queries.jsonl"
        doc_file_path = f"/home/work/retrieval/data/msmarco/{num}_docs.jsonl"
        doc_list = []
        with open(query_file_path, "r", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line.strip())
                doc_list.append(all_docs[record["qid"]])
        with open(doc_file_path, "w") as file:
            for result in doc_list:
                file.write(result + "\n")
        print(f"output: {doc_file_path}")


def generate_queries():
    for num in range(5):
        input_file = f"/home/work/retrieval/data/msmarco/queries_{num}.tsv"
        output_file = f"/home/work/retrieval/data/msmarco/{num}_queries.jsonl"

        with open(input_file, "r", encoding="utf-8") as tsv_file, open(
            output_file, "w", encoding="utf-8"
        ) as jsonl_file:
            tsv_reader = csv.reader(tsv_file, delimiter="\t")
            # 헤더 건너뛰기 (필요 없으면 제거)
            next(tsv_reader)
            # 각 행을 읽어 JSONL 형식으로 저장
            for row in tsv_reader:
                qid = row[0]
                query_text = row[1]
                json_obj = {"qid": qid, "query": query_text, "ans_pids": [f"d{qid}"]}
                jsonl_file.write(json.dumps(json_obj) + "\n")
        print(f"변환 완료: {output_file}")


if __name__ == "__main__":
    # generate_queries()
    generate_docs()
