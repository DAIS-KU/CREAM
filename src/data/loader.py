import json
import random

# writing_test_qas.forum-353, writing_test_collection-85121
domain_list = ["writing", "lifestyle", "technology", "science", "recreation"]
dataset_list = [
    "dev_qas.search",
    "dev_qas.forum",
    "test_qas.forum",
    "test_qas.search",
    "dev_collection",
    "test_collection",
]
prefix_map = {
    f"{d}_{ds}": i * 10 + j
    for i, d in enumerate(domain_list)
    for j, ds in enumerate(dataset_list)
}


def convert_str_id_to_number_id(str_id):
    parts = str_id.split("-")
    prefix_part, id_part = parts[0], parts[-1]
    prefix = prefix_map[prefix_part]
    number_id = int(f"{prefix}{id_part}")
    return number_id


def read_jsonl(file_path, is_query, as_number_id=False):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line.strip())
            if as_number_id:
                id_field = "qid" if is_query else "doc_id"
                key = record.get(id_field)
                if key is None:
                    raise ValueError("id field cannot be null")
                record[id_field] = convert_str_id_to_number_id(key)
                if is_query:
                    record["answer_pids"] = [
                        convert_str_id_to_number_id(pid)
                        for pid in record["answer_pids"]
                    ]
            data.append(record)
    return data


def count_jsonl_elements(file_path):
    count = 0
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():  # 빈 줄 무시
                count += 1
    return count


def read_jsonl_as_dict(file_path, id_field, as_number_id=False):
    data_dict = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line.strip())
            key = record.get(id_field)
            if key is None:
                raise ValueError("id field cannot be null")
            if as_number_id:
                key = convert_str_id_to_number_id(key)
                record[id_field] = key
            data_dict[key] = record
    return data_dict


def load_train_docs(session_number=None):
    train_docs = {}
    if session_number:
        print(f"Read {i}th docs")
        doc_path = f"../data/train_session{i}_docs.jsonl"
        doc_data = read_jsonl_as_dict(doc_path, "doc_id")
        train_docs.update(doc_data)
    else:
        for i in range(4):
            print(f"Read {i}th docs")
            doc_path = f"../data/train_session{i}_docs.jsonl"
            doc_data = read_jsonl_as_dict(doc_path, "doc_id")
            train_docs.update(doc_data)
    print(f"train doc size {len(train_docs)}")
    return train_docs


def load_eval_docs(session_number):
    eval_docs = []
    for i in range(session + 1):
        print(f"Read {i}th docs")
        doc_path = f"../data/train_session{i}_docs.jsonl"
        doc_data = read_jsonl(doc_path, False)
        eval_docs.extend(doc_data)
        doc_path = f"../data/test_session{i}_docs.jsonl"
        doc_data = read_jsonl(doc_path, False)
        eval_docs.extend(doc_data)
    return eval_docs


def sample_data(data, percentage):
    sample_size = int(len(data) * percentage)
    return random.sample(data, sample_size)


def save_jsonl(data, file_name, mode="w"):
    with open(file_name, mode, encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_to_jsonl(file_path, data):
    with open(file_path, "a") as f:
        json_line = json.dumps(data)
        f.write(json_line + "\n")


def write_file(rank_file_path, result):
    with open(rank_file_path, "w") as f:
        for key, values in result.items():
            line = f"{key} " + " ".join(map(str, values)) + "\n"
            f.write(line)


def write_lines(filename, result_list):
    with open(filename, "w") as file:
        for result in result_list:
            file.write(result + "\n")


def write_line(filename, result):
    with open(filename, "w") as file:
        file.write(result)
