import json
import os
import math


def generate_query_and_docs(input_file, queries_file, docs_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(queries_file, "w", encoding="utf-8") as qf, open(
        docs_file, "w", encoding="utf-8"
    ) as df:
        for item in data:
            query_id = item["query_id"]
            question = item["question"]
            context = item["context"]

            # queries.jsonl용 레코드
            query_obj = {
                "qid": query_id,
                "query": question,
                "answer_pids": [f"doc_{query_id}"],
            }
            qf.write(json.dumps(query_obj, ensure_ascii=False) + "\n")

            # docs.jsonl용 레코드
            doc_obj = {"doc_id": f"doc_{query_id}", "text": context}
            df.write(json.dumps(doc_obj, ensure_ascii=False) + "\n")

    print("✅ 변환 완료: queries.jsonl, docs.jsonl 생성됨")


def split_train_test(
    queries_file,
    docs_file,
    train_query_file,
    test_query_file,
    train_docs_file,
    test_docs_file,
):
    with open(queries_file, "r", encoding="utf-8") as qf:
        queries = [json.loads(line) for line in qf]

    with open(docs_file, "r", encoding="utf-8") as df:
        docs = [json.loads(line) for line in df]
    # 개수 일치 확인
    assert len(queries) == len(docs), "❌ queries와 docs 개수가 다릅니다!"
    # 9:1 비율로 순서 유지 분할
    with open(train_query_file, "w", encoding="utf-8") as tqf, open(
        test_query_file, "w", encoding="utf-8"
    ) as teqf, open(train_docs_file, "w", encoding="utf-8") as tdf, open(
        test_docs_file, "w", encoding="utf-8"
    ) as tedf:

        for i, (q, d) in enumerate(zip(queries, docs)):
            # 9개 train → 1개 test 주기 반복
            if i % 10 == 9:  # 10번째마다 test로
                teqf.write(json.dumps(q, ensure_ascii=False) + "\n")
                tedf.write(json.dumps(d, ensure_ascii=False) + "\n")
            else:
                tqf.write(json.dumps(q, ensure_ascii=False) + "\n")
                tdf.write(json.dumps(d, ensure_ascii=False) + "\n")


def split_sessions(
    train_query_file,
    test_query_file,
    train_docs_file,
    test_docs_file,
    base_dir="/home/work/.default/huijeong/cream/data/ChricinclingAmericaQA/splited",
):
    files = [train_query_file, test_query_file, train_docs_file, test_docs_file]
    for fname in files:
        path = os.path.join(base_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total = len(lines)
        chunk_size = math.ceil(total / 10)

        print(f"📄 {fname}: {total}개 항목 → {chunk_size}개씩 10분할")

        for i in range(10):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total)
            chunk_lines = lines[start:end]

            out_name = fname.replace(".jsonl", f"_{i}.jsonl")
            out_path = os.path.join(base_dir, out_name)

            with open(out_path, "w", encoding="utf-8") as out_f:
                out_f.writelines(chunk_lines)


def _evenly_sample_indices(n: int, k: int):
    """
    길이 n에서 k개를 '상대순서 유지'하며 균등 간격으로 샘플링할 인덱스 목록을 생성.
    - k >= n이면 0..n-1 전체 반환
    - k == 1이면 [0]
    - 그 외에는 round(i*(n-1)/(k-1)) 방식으로 등간격 선택 + 중복 제거
    """
    if n == 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [0]
    idxs = []
    prev = -1
    for i in range(k):
        # 등간격 위치를 반올림으로 선택
        idx = round(i * (n - 1) / (k - 1))
        if idx != prev:
            idxs.append(idx)
            prev = idx
    # 혹시 중복 제거 과정에서 k개보다 작아졌다면 끝에서 보충
    j = n - 1
    while len(idxs) < k and j > idxs[-1]:
        idxs.append(j)
        j -= 1
    return idxs[:k]


def sample_query_shards(
    src_base_dir: str,
    dst_base_dir: str,
    train_prefix: str = "train_query",
    test_prefix: str = "test_query",
    train_sampled_prefix: str = "train",
    test_sampled_prefix: str = "test",
    train_per_shard: int = 2430,
    test_per_shard: int = 270,
    num_shards: int = 10,
):
    """
    0~(num_shards-1)번 shard 파일에서
      - train: 각 shard에서 train_per_shard개
      - test:  각 shard에서 test_per_shard개
    를 '상대순서 유지'하며 균등 간격 샘플링하고,
    dst_base_dir 아래에 저장합니다.

    입력 파일명 형식: {train_prefix}_{i}.jsonl, {test_prefix}_{i}.jsonl
    출력 파일명 형식: {train_prefix}_sampled_{i}.jsonl, {test_prefix}_sampled_{i}.jsonl
    """

    os.makedirs(dst_base_dir, exist_ok=True)

    def process(prefix: str, sampled_prefix: str, per_shard: int):
        for i in range(num_shards):
            src_path = os.path.join(src_base_dir, f"{prefix}_{i}.jsonl")
            if not os.path.exists(src_path):
                print(f"⚠️  건너뜀: {src_path} (없음)")
                continue

            with open(src_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            n = len(lines)
            idxs = _evenly_sample_indices(n, per_shard)
            sampled = [lines[idx] for idx in idxs]

            out_path = os.path.join(
                dst_base_dir, f"{sampled_prefix}_session{i}_queries.jsonl"
            )
            with open(out_path, "w", encoding="utf-8") as out:
                out.writelines(sampled)

            print(
                f"✅ {prefix}_{i}.jsonl → {os.path.basename(out_path)} "
                f"({len(sampled)}/{n}개, 순서 유지 샘플링)"
            )

    # train / test 각각 처리
    process(train_prefix, train_sampled_prefix, train_per_shard)
    process(test_prefix, test_sampled_prefix, test_per_shard)


if __name__ == "__main__":
    # input_file = "/home/work/.default/huijeong/cream/data/ChricinclingAmericaQA/train.json"
    # queries_file = "/home/work/.default/huijeong/cream/data/news_query.jsonl"
    # docs_file = "/home/work/.default/huijeong/cream/data/news_docs.jsonl"
    # generate_query_and_docs(input_file, queries_file, docs_file)
    # split_train_test(
    #     queries_file = "/home/work/.default/huijeong/cream/data/ChricinclingAmericaQA/news_query.jsonl",
    #     docs_file = "/home/work/.default/huijeong/cream/data/ChricinclingAmericaQA/news_docs.jsonl",
    #     train_query_file="/home/work/.default/huijeong/cream/data/ChricinclingAmericaQA/train_query.jsonl",
    #     test_query_file="/home/work/.default/huijeong/cream/data/ChricinclingAmericaQA/test_query.jsonl",
    #     train_docs_file="/home/work/.default/huijeong/cream/data/ChricinclingAmericaQA/train_docs.jsonl",
    #     test_docs_file="/home/work/.default/huijeong/cream/data/ChricinclingAmericaQA/test_docs.jsonl" )
    # split_sessions(
    #     train_query_file="/home/work/.default/huijeong/cream/data/ChricinclingAmericaQA/train_query.jsonl",
    #     test_query_file="/home/work/.default/huijeong/cream/data/ChricinclingAmericaQA/test_query.jsonl",
    #     train_docs_file="/home/work/.default/huijeong/cream/data/ChricinclingAmericaQA/train_docs.jsonl",
    #     test_docs_file="/home/work/.default/huijeong/cream/data/ChricinclingAmericaQA/test_docs.jsonl")
    sample_query_shards(
        src_base_dir="/home/work/.default/huijeong/cream/data/ChricinclingAmericaQA/splited",
        dst_base_dir="/home/work/.default/huijeong/cream/data/datasetN_large",
    )
