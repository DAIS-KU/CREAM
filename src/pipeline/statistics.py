import json
from transformers import BertTokenizer


def load_texts(file_path, field_name="text"):
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                texts.append(obj[field_name])
    return texts


def summary(docs_path, query_path):
    # BERT 토크나이저 로드 (기본 uncased 버전 예시)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # 텍스트 불러오기
    docs_texts = load_texts(docs_path, "text")
    query_texts = load_texts(query_path, "query")
    all_texts = docs_texts + query_texts

    all_tokens = []
    for text in all_texts:
        tokens = tokenizer.tokenize(text)  # WordPiece 토큰화
        all_tokens.extend(tokens)

    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))

    print(f"전체 토큰 수 (BERT 기준): {total_tokens}")
    print(f"고유한 토큰 수 (BERT 기준): {unique_tokens}")


def get_summary():
    for i in range(10):
        docs_path = f"/home/work/.default/huijeong/data/lotte_session/train_session{i}_docs_filtered.jsonl"
        query_path = f"/home/work/.default/huijeong/data/lotte_session/train_session{i}_queries.jsonl"
        print(f'=========== Session {i} ===========')
        summary(docs_path, query_path)
