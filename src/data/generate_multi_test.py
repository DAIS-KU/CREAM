import json
import random
from collections import Counter, defaultdict

from datasets import load_dataset
import os

from loader import append_to_jsonl, read_jsonl, read_jsonl_as_dict


def msmarco_check_query_answer_docs(domains):
    for domain in domains:
        quries_path = f"./raw/merged/{domain}_queries.jsonl"
        queries = read_jsonl(quries_path)
        answer_doc_ids = set()
        total_answer_doc_cnt = 0
        for q in queries:
            answer_doc_ids.update(q["answer_pids"])
            total_answer_doc_cnt += len(q["answer_pids"])
        answer_cnt = len(answer_doc_ids)

        doc_path = f"./raw/merged/{domain}_docs.jsonl"
        doc_cnt = count_jsonl_elements(doc_path)

        print(
            f"{domain} : answer {answer_cnt} / total {doc_cnt} = {answer_cnt/doc_cnt * 100}%, total_answer_doc_cnt: {total_answer_doc_cnt}"
        )


def append_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def sessioning(
    domains,
    domain_answer_rate,
    need_train_query_counts,
    need_test_query_counts,
    dataset="msmarco",
):
    session_count = len(need_train_query_counts)
    print(f"Session count {session_count}")

    for domain, ans_rate in zip(domains, domain_answer_rate):
        queries = read_jsonl(
            f"/home/work/retrieval/data/raw/{dataset}/{domain}_queries.jsonl", True
        )
        random.shuffle(queries)

        answer_train_doc_ids = defaultdict(set)
        answer_test_doc_ids = defaultdict(set)

        train_cnts = [0] * session_count
        test_cnts = [0] * session_count
        train_query_batches = [[] for _ in range(session_count)]
        test_query_batches = [[] for _ in range(session_count)]

        for query in queries:
            assigned = False
            for i in range(session_count):
                if train_cnts[i] < need_train_query_counts[i]:
                    train_cnts[i] += 1
                    train_query_batches[i].append(query)
                    answer_train_doc_ids[i].update(query["answer_pids"])
                    assigned = True
                    break
            if not assigned:
                for i in range(session_count):
                    if test_cnts[i] < need_test_query_counts[i]:
                        test_cnts[i] += 1
                        test_query_batches[i].append(query)
                        answer_test_doc_ids[i].update(query["answer_pids"])
                        assigned = True
                        break
            if not assigned:
                break

        for i in range(session_count):
            append_jsonl(
                train_query_batches[i],
                f"/home/work/retrieval/data/datasetM/{dataset}/train_session{i}_queries.jsonl",
            )
            append_jsonl(
                test_query_batches[i],
                f"/home/work/retrieval/data/datasetM/{dataset}/test_session{i}_queries.jsonl",
            )

        print(f"[DONE] Train Query {domain} | {' / '.join(map(str, train_cnts))}")
        print(f"[DONE] Test Query {domain} | {' / '.join(map(str, test_cnts))}")

        docs = read_jsonl_as_dict(
            f"/home/work/retrieval/data/raw/{dataset}/{domain}_docs.jsonl",
            id_field="doc_id",
        )

        train_need_doc_counts = [
            len(answer_train_doc_ids[i]) / ans_rate * 100 for i in range(session_count)
        ]
        test_need_doc_counts = [
            len(answer_test_doc_ids[i]) / ans_rate * 100 for i in range(session_count)
        ]

        print(
            f"train_need_doc_counts: {train_need_doc_counts}, test_need_doc_counts: {test_need_doc_counts}"
        )

        train_cnts = [0] * session_count
        test_cnts = [0] * session_count
        train_doc_batches = [[] for _ in range(session_count)]
        test_doc_batches = [[] for _ in range(session_count)]

        for i in range(session_count):
            for doc_id in answer_train_doc_ids[i]:
                if doc_id in docs:
                    doc = docs[doc_id]
                    doc["text"] = doc["text"][:4096]
                    train_doc_batches[i].append(doc)
                    train_cnts[i] += 1
            for doc_id in answer_test_doc_ids[i]:
                if doc_id in docs:
                    doc = docs[doc_id]
                    doc["text"] = doc["text"][:4096]
                    test_doc_batches[i].append(doc)
                    test_cnts[i] += 1

        for i in range(session_count):
            append_jsonl(
                train_doc_batches[i],
                f"/home/work/retrieval/data/datasetM/{dataset}/train_session{i}_docs.jsonl",
            )
            append_jsonl(
                test_doc_batches[i],
                f"/home/work/retrieval/data/datasetM/{dataset}/test_session{i}_docs.jsonl",
            )

        print(
            f"[DONE] Train Document {domain} answer  {' / '.join(map(str, train_cnts))}"
        )
        print(
            f"[DONE] Test Document {domain} answer  {' / '.join(map(str, test_cnts))}"
        )

        used_doc_ids = set()
        for i in range(session_count):
            used_doc_ids.update(answer_train_doc_ids[i])
            used_doc_ids.update(answer_test_doc_ids[i])

        left_doc_ids = list(set(docs.keys()) - used_doc_ids)
        random.shuffle(left_doc_ids)

        left_train_doc_batches = [[] for _ in range(session_count)]
        left_test_doc_batches = [[] for _ in range(session_count)]

        for doc_cnt, doc_id in enumerate(left_doc_ids):
            assigned = False
            for i in range(session_count):
                if train_cnts[i] < train_need_doc_counts[i]:
                    doc = docs[doc_id]
                    doc["text"] = doc["text"][:4096]
                    left_train_doc_batches[i].append(doc)
                    train_cnts[i] += 1
                    assigned = True
                    break
            if not assigned:
                for i in range(session_count):
                    if test_cnts[i] < test_need_doc_counts[i]:
                        doc = docs[doc_id]
                        doc["text"] = doc["text"][:4096]
                        left_test_doc_batches[i].append(doc)
                        test_cnts[i] += 1
                        assigned = True
                        break
            if not assigned:
                break
            if doc_cnt % 1000 == 0:
                print(
                    f"Document Train {domain} no-answer | {' / '.join(map(str, train_cnts))}"
                )
                print(
                    f"Document Test {domain} no-answer | {' / '.join(map(str, test_cnts))}"
                )

        for i in range(session_count):
            append_jsonl(
                left_train_doc_batches[i],
                f"/home/work/retrieval/data/datasetM/{dataset}/train_session{i}_docs.jsonl",
            )
            append_jsonl(
                left_test_doc_batches[i],
                f"/home/work/retrieval/data/datasetM/{dataset}/test_session{i}_docs.jsonl",
            )

        print(
            f"[DONE] Document Train {domain} no-answer | {' / '.join(map(str, train_cnts))}"
        )
        print(
            f"[DONE] Document Test {domain} no-answer | {' / '.join(map(str, test_cnts))}"
        )


def sessioning_accumulate(
    domains,
    domain_answer_rate,
    need_train_query_counts,
    need_test_query_counts,
    dataset="msmarco",
):
    session_count = len(need_train_query_counts)
    print(f"Session count {session_count}")

    for domain, ans_rate in zip(domains, domain_answer_rate):
        # 1) 쿼리 로드 및 셔플
        queries = read_jsonl(
            f"/home/work/retrieval/data/raw/{dataset}/{domain}_queries.jsonl", True
        )
        random.shuffle(queries)

        # 2) 세션별 ID/카운트/배치 초기화
        answer_train_doc_ids = defaultdict(set)
        answer_test_doc_ids = defaultdict(set)
        train_cnts = [0] * session_count
        test_cnts = [0] * session_count
        train_q_batches = [[] for _ in range(session_count)]
        test_q_batches = [[] for _ in range(session_count)]

        # 3) 쿼리 분배
        for q in queries:
            placed = False
            # ▶ 학습
            for i in range(session_count):
                if train_cnts[i] < need_train_query_counts[i]:
                    train_cnts[i] += 1
                    train_q_batches[i].append(q)
                    answer_train_doc_ids[i].update(q["answer_pids"])
                    placed = True
                    break
            if placed:
                continue
            # ▶ 테스트
            for i in range(session_count):
                if test_cnts[i] < need_test_query_counts[i]:
                    test_cnts[i] += 1
                    test_q_batches[i].append(q)
                    answer_test_doc_ids[i].update(q["answer_pids"])
                    placed = True
                    break
            if not placed:
                break

        # 4) 학습 쿼리 저장
        for i in range(session_count):
            append_jsonl(
                train_q_batches[i],
                f"/home/work/retrieval/data/datasetM/{dataset}/train_session{i}_queries.jsonl",
            )
        print(f"[DONE] Train Query {domain} | {'/'.join(map(str, train_cnts))}")
        print(f"[DONE] Test  Query {domain} | {'/'.join(map(str, test_cnts))}")

        # 5) 문서 로드
        docs = read_jsonl_as_dict(
            f"/home/work/retrieval/data/raw/{dataset}/{domain}_docs.jsonl",
            id_field="doc_id",
        )

        # 6) 필요한 문서 수 계산
        train_need_docs = [
            len(answer_train_doc_ids[i]) / ans_rate * 100 for i in range(session_count)
        ]
        test_need_docs = [
            len(answer_test_doc_ids[i]) / ans_rate * 100 for i in range(session_count)
        ]
        print(f"train_need_doc_counts: {train_need_docs}")
        print(f"test_need_doc_counts:  {test_need_docs}")

        # 7) 세션별 문서 배치 초기화
        train_cnts = [0] * session_count
        test_cnts = [0] * session_count
        train_d_batches = [[] for _ in range(session_count)]
        test_d_batches = [[] for _ in range(session_count)]
        left_train_d_batches = [[] for _ in range(session_count)]
        left_test_d_batches = [[] for _ in range(session_count)]

        # 8) 정답 문서(학습/테스트) 배치
        for i in range(session_count):
            for doc_id in answer_train_doc_ids[i]:
                if doc_id in docs:
                    d = docs[doc_id]
                    d["text"] = d["text"][:4096]
                    train_d_batches[i].append(d)
                    train_cnts[i] += 1
            for doc_id in answer_test_doc_ids[i]:
                if doc_id in docs:
                    d = docs[doc_id]
                    d["text"] = d["text"][:4096]
                    test_d_batches[i].append(d)
                    test_cnts[i] += 1

        # 9) 학습 문서 저장
        for i in range(session_count):
            append_jsonl(
                train_d_batches[i],
                f"/home/work/retrieval/data/datasetM/{dataset}/train_session{i}_docs.jsonl",
            )
        print(f"[DONE] Train Docs   {domain} | {'/'.join(map(str, train_cnts))}")
        print(f"[DONE] Test  Docs   {domain} | {'/'.join(map(str, test_cnts))}")

        # 10) 남은 문서(no-answer) 분배
        used_ids = set().union(
            *answer_train_doc_ids.values(), *answer_test_doc_ids.values()
        )
        leftover = list(set(docs.keys()) - used_ids)
        random.shuffle(leftover)

        for cnt, doc_id in enumerate(leftover):
            placed = False
            for i in range(session_count):
                if train_cnts[i] < train_need_docs[i]:
                    d = docs[doc_id]
                    d["text"] = d["text"][:4096]
                    left_train_d_batches[i].append(d)
                    train_cnts[i] += 1
                    placed = True
                    break
            if placed:
                continue
            for i in range(session_count):
                if test_cnts[i] < test_need_docs[i]:
                    d = docs[doc_id]
                    d["text"] = d["text"][:4096]
                    left_test_d_batches[i].append(d)
                    test_cnts[i] += 1
                    placed = True
                    break
            if not placed:
                break
            if cnt % 1000 == 0:
                print(f"Leftover train | {'/'.join(map(str, train_cnts))}")
                print(f"Leftover test  | {'/'.join(map(str, test_cnts))}")

        # 11) 남은 학습 문서 저장
        for i in range(session_count):
            append_jsonl(
                left_train_d_batches[i],
                f"/home/work/retrieval/data/datasetM/{dataset}/train_session{i}_docs.jsonl",
            )
        print(f"[DONE] Leftover Train Docs {domain}")

        # 12) 누적 테스트 쿼리 및 문서셋 저장
        cumulative_qs = []
        cumulative_docs = []
        for i in range(session_count):
            # ▶ 쿼리 누적
            cumulative_qs.extend(test_q_batches[i])
            append_jsonl(
                cumulative_qs,
                f"/home/work/retrieval/data/datasetM/{dataset}/test_session{i}_queries.jsonl",
            )
            # ▶ 문서 누적 (정답 + no-answer)
            cumulative_docs.extend(test_d_batches[i])
            cumulative_docs.extend(left_test_d_batches[i])
            append_jsonl(
                cumulative_docs,
                f"/home/work/retrieval/data/datasetM/{dataset}/test_session{i}_docs.jsonl",
            )
        print(f"[DONE] Cumulative Test Queries & Docs {domain}")


def split_data(queries, documents, num_splits=3, queries_per_prefix=13):
    doc_dict = {doc["doc_id"]: doc for doc in documents}
    query_prefix_groups = defaultdict(list)

    for query in queries:
        prefix = query["qid"].split("_")[0]
        query_prefix_groups[prefix].append(query)

    query_subsets = [[] for _ in range(num_splits)]
    doc_subsets = [set() for _ in range(num_splits)]

    for prefix, query_list in query_prefix_groups.items():
        random.shuffle(query_list)
        prefix_splits = [
            query_list[i : i + queries_per_prefix]
            for i in range(0, len(query_list), queries_per_prefix)
        ]

        for i, split in enumerate(prefix_splits):
            subset_idx = i % num_splits
            query_subsets[subset_idx].extend(split)
            for query in split:
                doc_subsets[subset_idx].update(query["answer_pids"])
    print(f"Query Subsets Distribution: {[len(q) for q in query_subsets]}")
    doc_subsets = [
        [doc_dict[pid] for pid in doc_set if pid in doc_dict] for doc_set in doc_subsets
    ]

    remaining_docs = set(doc_dict.keys()) - set().union(
        *[set(d["doc_id"] for d in subset) for subset in doc_subsets]
    )
    remaining_docs = list(remaining_docs)
    random.shuffle(remaining_docs)

    for i, doc_id in enumerate(remaining_docs):
        doc_subsets[i % num_splits].append(doc_dict[doc_id])

    doc_subsets = [list(subset) for subset in doc_subsets]

    def print_distribution(subsets, name, _id):
        for i, subset in enumerate(subsets):
            total_count = len(subset)
            if total_count == 0:
                print(f"{name} Subset {i+1} Distribution: Empty")
                continue

            prefix_counts = Counter([item[_id].split("_")[0] for item in subset])
            prefix_ratios = {
                prefix: round((count / total_count) * 100, 2)
                for prefix, count in prefix_counts.items()
            }

            sorted_ratios = {
                key: prefix_ratios[key] for key in sorted(prefix_ratios.keys())
            }

            print(
                f"{name} Subset {i+1} Distribution: {sorted_ratios} (Total: {total_count})"
            )

    print_distribution(query_subsets, "Queries", "qid")
    print_distribution(doc_subsets, "Documents", "doc_id")
    return query_subsets, doc_subsets


def read_jsonl_line(filepath):
    res = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            res.append(json.loads(line.strip()))
    return res


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def process_hotpot_qa(path="../data/hotpot_dev_distractor_v1.json"):
    datas = read_jsonl_line(path)[0]
    processed = []
    for i, data in enumerate(datas):
        question = data["question"]
        golden_answer_parts = [data["answer"]]
        for fact in data["supporting_facts"]:
            context_name, sentence_idx = fact
            for context_item in data["context"]:
                if context_item[0] == context_name:
                    if sentence_idx < len(context_item[1]):
                        golden_answer_parts.append(context_item[1][sentence_idx])
                    else:
                        print(
                            f"Warning: Index {sentence_idx} out of range for context '{context_name}'"
                        )
                        continue
        golden_answer = " ".join(golden_answer_parts)
        qna = {"query": question, "answer": golden_answer}
        processed.append(qna)
        if i % 1000 == 0:
            print(f"{i}th query and answer: {qna}")
        if i == 13000:
            break
    save_jsonl(processed, "/home/work/retrieval/data/raw/msmarco/pretrained,jsonl")


def check_duplicate(session_count=12):
    doc_id_to_files = defaultdict(set)
    for i in range(session_count):
        src_docs_path = (
            f"/home/work/retrieval/data/raw/msmarco/train_session{i}_docs.jsonl"
        )
        with open(src_docs_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                doc_id = obj.get("doc_id")
                if doc_id is not None:
                    doc_id_to_files[doc_id].add(src_docs_path)
    duplicateds = {
        doc_id: sorted(list(files))
        for doc_id, files in doc_id_to_files.items()
        if len(files) > 1
    }
    for doc_id, files in list(duplicateds.items()):
        print(f"{doc_id} | {files}")


if __name__ == "__main__":

    sessioning(
        dataset="msmarco",
        domains=[
            "domain0",
        ],  # 'lifestyle',  'science', 'recreation', 'lifestyle'
        domain_answer_rate=[4.27],  # 1.82, 6.47, 1.47, 5.17, 6.43
        need_train_query_counts=[
            150,
            150,
            0,
            0,
            0,
            0,
            150,
            150,
            0,
        ],  # [0,0,0,300,0 ,0,0,0,300,0, 0,0,0,300,0],
        need_test_query_counts=[
            20,
            20,
            20,
            0,
            0,
            0,
            20,
            20,
            20,
        ],  # [0,0,0,20,20, 0,0,0,20,20, 0,0,0,20,20],
    )

    sessioning(
        dataset="msmarco",
        domains=[
            "domain1",
        ],  # 'lifestyle',  'science', 'recreation', 'lifestyle'
        domain_answer_rate=[4.27],  # 1.82, 6.47, 1.47, 5.17, 6.43
        need_train_query_counts=[
            0,
            0,
            0,
            150,
            150,
            0,
            0,
            150,
            150,
        ],  # [300,0,0,0,0, 300,0,0,0,0, 300,0,0,0,0],
        need_test_query_counts=[
            0,
            0,
            0,
            20,
            20,
            20,
            0,
            20,
            20,
        ],  # [20,20,0,0,0,20,20,0,0,0, 20,20,0,0,0],
    )

    sessioning(
        dataset="msmarco",
        domains=[
            "domain2",
        ],  # 'lifestyle',  'science', 'recreation', 'lifestyle'
        domain_answer_rate=[4.27],  # 1.82, 6.47, 1.47, 5.17, 6.43
        need_train_query_counts=[
            0,
            150,
            150,
            0,
            0,
            150,
            150,
            0,
            0,
        ],  # [0,0,300,0,0, 0,0,300,0,0, 0,0,300,0,0],
        need_test_query_counts=[
            0,
            20,
            20,
            20,
            0,
            20,
            20,
            20,
            0,
        ],  # [0,0,20,20,0, 0,0,20,20,0, 0,0,20,20,0],
    )
    sessioning(
        dataset="msmarco",
        domains=[
            "domain3",
        ],  # 'lifestyle',  'science', 'recreation', 'lifestyle'
        domain_answer_rate=[4.27],  # 1.82, 6.47, 1.47, 5.17, 6.43
        need_train_query_counts=[
            150,
            0,
            0,
            0,
            150,
            150,
            0,
            0,
            150,
        ],  # [0,300,0,0,0,0,300,0,0,0, 0,300,0,0,0],
        need_test_query_counts=[
            20,
            20,
            0,
            0,
            20,
            20,
            20,
            0,
            20,
        ],  # [0,20,20,0,0, 0,20,20,0,0, 0,20,20,0,0],
    )

    sessioning(
        dataset="msmarco",
        domains=[
            "domain4",
        ],  # 'lifestyle',  'science', 'recreation', 'lifestyle'
        domain_answer_rate=[4.27],  # 1.82, 6.47, 1.47, 5.17, 6.43
        need_train_query_counts=[
            0,
            0,
            150,
            150,
            0,
            0,
            0,
            0,
            0,
        ],  # [0,0,0,0,300, 0,0,0,0,300, 0,0,0,0,300],
        need_test_query_counts=[
            0,
            0,
            20,
            20,
            20,
            0,
            0,
            0,
            0,
        ],  # [0,0,0,0,20, 20,0,0,0,20, 20,0,0,20,20],
    )
