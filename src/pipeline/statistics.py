import json


def average_field_length(file_path, field_name="text"):
    total_chars = 0
    total_entries = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            value = data.get(field_name, "")

            if isinstance(value, str):
                total_chars += len(value)
                total_entries += 1

    if total_entries == 0:
        return 0
    print(f"total_entries: {total_entries}")
    return total_chars / total_entries
