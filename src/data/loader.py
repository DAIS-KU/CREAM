import json
import random


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def count_jsonl_elements(file_path):
    count = 0
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():  # 빈 줄 무시
                count += 1
    return count


def read_jsonl_as_dict(file_path, id_field):
    data_dict = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line.strip())
            key = record.get(id_field)  # id_field로 키를 가져옴
            if key is not None:
                data_dict[key] = record  # id를 키로, 나머지 데이터를 값으로 저장
    return data_dict


def sample_data(data, percentage):
    sample_size = int(len(data) * percentage)
    return random.sample(data, sample_size)


def save_jsonl(data, file_name):
    with open(file_name, "w", encoding="utf-8") as file:
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
