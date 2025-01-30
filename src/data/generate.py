from . import *
from collections import defaultdict


def sessioning(domains, query_sample_rate, doc_sample_rate, session_ratio, k):
    for domain in domains:
        train_quries_path = f"./raw/merged/{domain}_queries.jsonl"
        train_queries = read_jsonl(train_quries_path)
        random.shuffle(train_queries)
        total_query_cnt = len(train_queries)
        answer_doc_ids = defaultdict(set)
        exclude_doc_ids = set()

        need_query_counts = [
            k,
            total_query_cnt * 0.2,
            0,
            0,
        ]  # [round(total_query_cnt * ratio * query_sample_rate) for ratio in session_ratio]
        print(f"need_query_counts: {need_query_counts}")
        cnt0, cnt1, cnt2, cnt3 = 0, 0, 0, 0
        for query in train_queries:
            if cnt0 < need_query_counts[0]:
                dest_path = f"./raw/idea_validation/train{k}_{domain}_queries.jsonl"
                cnt0 += 1
                answer_doc_ids[0].update(query["answer_pids"])
            elif cnt1 < need_query_counts[1]:
                dest_path = f"./raw/idea_validation/eval{k}_{domain}_queries.jsonl"
                cnt1 += 1
                # 전체 정답 제외
                exclude_doc_ids.update(query["answer_pids"])
                # 포함 정답 3개 선정
                query["answer_pids"] = random.sample(
                    query["answer_pids"], min(len(query["answer_pids"]), 3)
                )
                # 평가문서셋에 포함시킬 정답 업데이트
                answer_doc_ids[1].update(query["answer_pids"])
            elif cnt2 < need_query_counts[2]:
                dest_path = f"./raw/idea_validation/train_queries.jsonl"
                cnt2 += 1
                answer_doc_ids[2].update(query["answer_pids"])
            else:
                break
            append_to_jsonl(dest_path, query)
        print(f"[DONE] Query {domain} | {cnt0} / {cnt1} / {cnt2} / {cnt3}")

        train_doc_path = f"./raw/merged/{domain}_docs.jsonl"
        train_docs = read_jsonl_as_dict(train_doc_path, id_field="doc_id")
        total_doc_cnt = len(train_docs)
        need_doc_counts = [
            0,
            100000,
            0,
            0,
        ]  # [round(total_doc_cnt * ratio * doc_sample_rate) for ratio in session_ratio]
        print(f"need_doc_counts: {need_doc_counts}")

        cnt0, cnt1, cnt2, cnt3 = 0, 0, 0, 0
        for doc_id in answer_doc_ids[0]:
            dest_path = f"./raw/idea_validation/train{k}_{domain}_docs.jsonl"
            cnt0 += 1
            append_to_jsonl(dest_path, train_docs[doc_id])

        # 정답 3개 반드시 포함
        for doc_id in answer_doc_ids[1]:
            dest_path = f"./raw/idea_validation/eval{k}_{domain}_docs.jsonl"
            cnt1 += 1
            append_to_jsonl(dest_path, train_docs[doc_id])

        for doc_id in answer_doc_ids[2]:
            dest_path = f"./raw/idea_validation/train_docs.jsonl"
            cnt2 += 1
            append_to_jsonl(dest_path, train_docs[doc_id])
        print(f"[DONE] Document {domain} answer  {cnt0} / {cnt1} / {cnt2} / {cnt3}")

        # 평가에 사용한 전체 정답 외 문서로 평가셋 채우기
        left_doc_ids = (
            set(train_docs.keys()) - exclude_doc_ids
        )  # - answer_doc_ids[0]- answer_doc_ids[1]- answer_doc_ids[2]
        left_doc_ids = list(left_doc_ids)
        random.shuffle(left_doc_ids)
        for doc_cnt, doc_id in enumerate(left_doc_ids):
            if cnt0 < need_doc_counts[0]:
                dest_path = f"./raw/idea_validation/train{k}_{domain}_docs.jsonl"
                cnt0 += 1
            elif cnt1 < need_doc_counts[1]:
                dest_path = f"./raw/idea_validation/eval{k}_{domain}_docs.jsonl"
                cnt1 += 1
            elif cnt2 < need_doc_counts[2]:
                dest_path = f"./raw/idea_validation/test_docs.jsonl"
                cnt2 += 1
            else:
                break
            append_to_jsonl(dest_path, train_docs[doc_id])
            if doc_cnt % 1000 == 0:
                print(f"Document {domain} no-answer | {cnt0}/{cnt1}/{cnt2}/{cnt3}")
        print(f"[DONE] Document {domain} no-answer | {cnt0}/{cnt1}/{cnt2}/{cnt3}")
