from collections import defaultdict

from . import *


def check_query_answer_docs(domains):
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


def exp_sessioning(
    domains, domain_answer_rate, need_train_query_counts, need_test_query_counts
):
    for domain, ans_rate in zip(domains, domain_answer_rate):
        queries = read_jsonl(f"./raw/merged/{domain}_queries.jsonl")
        random.shuffle(queries)
        answer_train_doc_ids, answer_test_doc_ids = defaultdict(set), defaultdict(set)

        print(
            f"need_train_query_counts: {need_train_query_counts}, need_test_query_counts: {need_test_query_counts}"
        )
        train_cnt0, train_cnt1, train_cnt2, train_cnt3 = 0, 0, 0, 0
        test_cnt0, test_cnt1, test_cnt2, test_cnt3 = 0, 0, 0, 0
        for query in queries:
            # TRAIN
            if train_cnt0 < need_train_query_counts[0]:
                dest_path = f"./raw/idea_validation/train_session0_queries.jsonl"
                train_cnt0 += 1
                answer_train_doc_ids[0].update(query["answer_pids"])
            elif train_cnt1 < need_train_query_counts[1]:
                dest_path = f"./raw/idea_validation/train_session1_queries.jsonl"
                train_cnt1 += 1
                answer_train_doc_ids[1].update(query["answer_pids"])
            elif train_cnt2 < need_train_query_counts[2]:
                dest_path = f"./raw/idea_validation/train_session2_queries.jsonl"
                train_cnt2 += 1
                answer_train_doc_ids[2].update(query["answer_pids"])
            elif train_cnt3 < need_train_query_counts[3]:
                dest_path = f"./raw/idea_validation/train_session3_queries.jsonl"
                train_cnt3 += 1
                answer_train_doc_ids[3].update(query["answer_pids"])
            # TEST
            elif test_cnt0 < need_test_query_counts[0]:
                dest_path = f"./raw/idea_validation/test_session0_queries.jsonl"
                test_cnt0 += 1
                answer_test_doc_ids[0].update(query["answer_pids"])
            elif test_cnt1 < need_test_query_counts[1]:
                dest_path = f"./raw/idea_validation/test_session1_queries.jsonl"
                test_cnt1 += 1
                answer_test_doc_ids[1].update(query["answer_pids"])
            elif test_cnt2 < need_test_query_counts[2]:
                dest_path = f"./raw/idea_validation/test_session2_queries.jsonl"
                test_cnt2 += 1
                answer_test_doc_ids[2].update(query["answer_pids"])
            elif test_cnt3 < need_test_query_counts[3]:
                dest_path = f"./raw/idea_validation/test_session3_queries.jsonl"
                test_cnt3 += 1
                answer_test_doc_ids[3].update(query["answer_pids"])
            else:
                break
            append_to_jsonl(dest_path, query)
        print(
            f"[DONE] Train Query {domain} | {train_cnt0} / {train_cnt1} / {train_cnt2} / {train_cnt3}"
        )
        print(
            f"[DONE] Test Query {domain} | {test_cnt0} / {test_cnt1} / {test_cnt2} / {test_cnt3}"
        )

        doc_path = f"./raw/merged/{domain}_docs.jsonl"
        docs = read_jsonl_as_dict(doc_path, id_field="doc_id")
        train_need_doc_counts = [
            len(answer_train_doc_ids[0]) / ans_rate,
            len(answer_train_doc_ids[1]) / ans_rate,
            len(answer_train_doc_ids[2]) / ans_rate,
            len(answer_train_doc_ids[3]) / ans_rate,
        ]
        test_need_doc_counts = [
            len(answer_test_doc_ids[0]) / ans_rate,
            len(answer_test_doc_ids[1]) / ans_rate,
            len(answer_test_doc_ids[2]) / ans_rate,
            len(answer_test_doc_ids[3]) / ans_rate,
        ]
        print(
            f"train_need_doc_counts: {train_need_doc_counts}, test_need_doc_counts: {test_need_doc_counts}"
        )

        # TRAIN
        train_cnt0, train_cnt1, train_cnt2, train_cnt3 = 0, 0, 0, 0
        for doc_id in answer_train_doc_ids[0]:
            dest_path = f"./raw/idea_validation/train_session0_docs.jsonl"
            train_cnt0 += 1
            append_to_jsonl(dest_path, docs[doc_id])
        for doc_id in answer_train_doc_ids[1]:
            dest_path = f"./raw/idea_validation/train_session1_docs.jsonl"
            train_cnt1 += 1
            append_to_jsonl(dest_path, docs[doc_id])
        for doc_id in answer_train_doc_ids[2]:
            dest_path = f"./raw/idea_validation/train_session2_docs.jsonl"
            train_cnt2 += 1
            append_to_jsonl(dest_path, docs[doc_id])
        for doc_id in answer_train_doc_ids[3]:
            dest_path = f"./raw/idea_validation/train_session3_docs.jsonl"
            train_cnt3 += 1
            append_to_jsonl(dest_path, docs[doc_id])
        print(
            f"[DONE] Train Document {domain} answer  {train_cnt0} / {train_cnt1} / {train_cnt2} / {train_cnt3}"
        )
        # TEST
        test_cnt0, test_cnt1, test_cnt2, test_cnt3 = 0, 0, 0, 0
        for doc_id in answer_test_doc_ids[0]:
            dest_path = f"./raw/idea_validation/test_session0_docs.jsonl"
            test_cnt0 += 1
            append_to_jsonl(dest_path, docs[doc_id])
        for doc_id in answer_test_doc_ids[1]:
            dest_path = f"./raw/idea_validation/test_session1_docs.jsonl"
            test_cnt1 += 1
            append_to_jsonl(dest_path, docs[doc_id])
        for doc_id in answer_test_doc_ids[2]:
            dest_path = f"./raw/idea_validation/test_session2_docs.jsonl"
            test_cnt2 += 1
            append_to_jsonl(dest_path, docs[doc_id])
        for doc_id in answer_test_doc_ids[3]:
            dest_path = f"./raw/idea_validation/test_session3_docs.jsonl"
            test_cnt3 += 1
            append_to_jsonl(dest_path, docs[doc_id])
        print(
            f"[DONE] Test Document {domain} answer  {test_cnt0} / {test_cnt1} / {test_cnt2} / {test_cnt3}"
        )

        left_doc_ids = (
            set(docs.keys())
            - answer_train_doc_ids[0]
            - answer_train_doc_ids[1]
            - answer_train_doc_ids[2]
            - answer_train_doc_ids[3]
            - answer_test_doc_ids[0]
            - answer_test_doc_ids[1]
            - answer_test_doc_ids[2]
            - answer_test_doc_ids[3]
        )
        left_doc_ids = list(left_doc_ids)
        random.shuffle(left_doc_ids)
        for doc_cnt, doc_id in enumerate(left_doc_ids):
            # TRAIN
            if train_cnt0 < train_need_doc_counts[0]:
                dest_path = f"./raw/idea_validation/train_session0_docs.jsonl"
                train_cnt0 += 1
            elif train_cnt1 < train_need_doc_counts[1]:
                dest_path = f"./raw/idea_validation/train_session1_docs.jsonl"
                train_cnt1 += 1
            elif train_cnt2 < train_need_doc_counts[2]:
                dest_path = f"./raw/idea_validation/train_session2_docs.jsonl"
                train_cnt2 += 1
            elif train_cnt3 < train_need_doc_counts[3]:
                dest_path = f"./raw/idea_validation/train_session3_docs.jsonl"
                train_cnt3 += 1
            # TEST
            elif test_cnt0 < test_need_doc_counts[0]:
                dest_path = f"./raw/idea_validation/test_session0_docs.jsonl"
                test_cnt0 += 1
            elif test_cnt1 < test_need_doc_counts[1]:
                dest_path = f"./raw/idea_validation/test_session1_docs.jsonl"
                test_cnt1 += 1
            elif test_cnt2 < test_need_doc_counts[2]:
                dest_path = f"./raw/idea_validation/test_session2_docs.jsonl"
                test_cnt2 += 1
            elif test_cnt3 < test_need_doc_counts[3]:
                dest_path = f"./raw/idea_validation/test_session3_docs.jsonl"
                test_cnt3 += 1
            else:
                break
            append_to_jsonl(dest_path, docs[doc_id])
            if doc_cnt % 1000 == 0:
                print(
                    f"Document Train {domain} no-answer | {train_cnt0}/{train_cnt1}/{train_cnt2}/{train_cnt3}"
                )
                print(
                    f"Document Test {domain} no-answer | {test_cnt0}/{test_cnt1}/{test_cnt2}/{test_cnt3}"
                )
        print(
            f"[DONE] Document Train {domain} no-answer | {train_cnt0}/{train_cnt1}/{train_cnt2}/{train_cnt3}"
        )
        print(
            f"[DONE] Document Test {domain} no-answer | {test_cnt0}/{test_cnt1}/{test_cnt2}/{test_cnt3}"
        )
