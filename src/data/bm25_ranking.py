from loader import read_jsonl, read_jsonl_as_dict, write_file, write_lines
from bm25 import BM25Okapi
from collections import defaultdict


def evaluate_dataset(query_path, rankings_path, doc_count, eval_log_path=None):
    eval_queries = read_jsonl(query_path, True)
    rankings = defaultdict(list)
    with open(rankings_path, "r") as f:
        for line in f:
            items = line.strip().split()
            qid: str = items[0]
            pids: List[str] = list(map(str, items[1:]))
            rankings[qid].extend(pids)

    top_k_success, top_k_recall = 5, 10
    domain_prefixes = [
        "writing",
        "technology",
        "lifestyle",
        "science",
        "recreation",
        "domain0",
        "domain1",
        "domain2",
        "domain3",
        "domain4",
    ]
    domain_stats = {
        prefix: {"success": 0, "recall": 0.0, "total": 0, "answer_count": 0}
        for prefix in domain_prefixes
    }
    overall_success = 0
    overall_recall = 0.0
    overall_total = 0
    total_answer_count = 0

    for query in eval_queries:
        qid = query["qid"]
        answer_pids = query["answer_pids"]
        total_answer_count += len(answer_pids)
        overall_total += 1

        hit_success = set(rankings[qid][:top_k_success]).intersection(answer_pids)
        hit_recall = set(rankings[qid][:top_k_recall]).intersection(answer_pids)

        if len(hit_success) > 0:
            overall_success += 1
        overall_recall += len(hit_recall) / len(answer_pids)

        for prefix in domain_prefixes:
            if qid.startswith(prefix):
                domain_stats[prefix]["total"] += 1
                domain_stats[prefix]["answer_count"] += len(answer_pids)
                if len(hit_success) > 0:
                    domain_stats[prefix]["success"] += 1
                domain_stats[prefix]["recall"] += len(hit_recall) / len(answer_pids)
                break

    res_strs = []
    res_strs.append(
        f"# query: {overall_total} | answers: {total_answer_count} | proportion: {total_answer_count / doc_count * 100:.1f}%"
    )
    res_strs.append(
        f"Avg Success@{top_k_success}: {overall_success / overall_total * 100:.1f}%"
    )
    res_strs.append(
        f"Avg Recall@{top_k_recall}: {overall_recall / overall_total * 100:.1f}%"
    )
    res_strs.append("")

    for prefix in domain_prefixes:
        stats = domain_stats[prefix]
        if stats["total"] > 0:
            res_strs.append(f"[{prefix}]")
            res_strs.append(
                f"  # query: {stats['total']} | answers: {stats['answer_count']} | proportion: {stats['answer_count'] / doc_count * 100:.1f}%"
            )
            res_strs.append(
                f"  Avg Success@{top_k_success}: {stats['success'] / stats['total'] * 100:.1f}%"
            )
            res_strs.append(
                f"  Avg Recall@{top_k_recall}: {stats['recall'] / stats['total'] * 100:.1f}%"
            )
            res_strs.append("")

    print("\n".join(res_strs))
    if eval_log_path:
        write_lines(eval_log_path, res_strs)


def preprocess(text):
    tokens = text.split(" ")
    # text = ''.join([char for char in text if char not in string.punctuation])
    # tokens = word_tokenize(text)
    # max_len = min(512, len(tokens))
    # tokens= tokens[:max_len]
    return tokens


def get_bm25(documents):
    doc_ids = [doc["doc_id"] for doc in documents]
    doc_texts = [doc["text"] for doc in documents]
    processed_docs = [preprocess(doc_text) for doc_text in doc_texts]

    bm25 = BM25Okapi(corpus=processed_docs)
    return bm25, doc_ids


def get_top_k_documents(query, bm25, doc_ids, k=10):
    query_tokens = preprocess(query["query"])
    scores = bm25.get_scores(query_tokens)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        :k
    ]
    top_k_doc_ids = [doc_ids[i] for i in top_k_indices]
    return top_k_doc_ids


# def evaluate_dataset(query_path, rankings_path, doc_count, eval_log_path=None):
#     eval_queries = read_jsonl(query_path, True)

#     rankings = defaultdict(list)
#     with open(rankings_path, "r") as f:
#         for line in f:
#             items = line.strip().split()
#             qid: str = items[0]
#             pids: List[str] = list(map(str, items[1:]))
#             rankings[qid].extend(pids)

#     top_k_success, top_k_recall = 5, 10
#     success = 0
#     num_q = 0
#     recall = 0.0
#     total_answer_count = 0

#     for query in eval_queries:
#         num_q += 1
#         qid = query["qid"]
#         answer_pids = query["answer_pids"]
#         total_answer_count += len(answer_pids)

#         hit_success = set(rankings[qid][:top_k_success]).intersection(answer_pids)
#         if len(hit_success) > 0:
#             success += 1

#         hit_recall = set(rankings[qid][:top_k_recall]).intersection(answer_pids)
#         recall += len(hit_recall) / len(answer_pids)

#     res_strs = [
#         f"# query:  {num_q} |  answers:{total_answer_count} | proportion:{total_answer_count/doc_count * 100:.1f} ",
#         f"Avg Success@{top_k_success}: {success / num_q * 100:.1f} ",
#         f"Avg Recall@{top_k_recall}: {recall / num_q * 100:.1f} ",
#     ]
#     print("".join(res_strs))
#     if eval_log_path:
#         write_lines(eval_log_path, res_strs)


def do_expermient(session_number=0):
    query_path = (
        f"/home/work/retrieval/data/datasetM/test_session{session_number}_queries.jsonl"
    )
    query_data = read_jsonl(query_path, True)

    eval_docs = []
    doc_path = (
        f"/home/work/retrieval/data/datasetM/test_session{session_number}_docs.jsonl"
    )
    doc_data = read_jsonl(doc_path, False)
    eval_docs.extend(doc_data)

    query_count = len(query_data)
    eval_doc_count = len(eval_docs)
    print(f"Query count:{query_count}, Document count:{eval_doc_count}")

    bm25, doc_ids = get_bm25(eval_docs)
    qcnt = 0
    result = defaultdict(list)

    for qidx in range(query_count):
        query = query_data[qidx]
        qid = query["qid"]
        top_k_doc_ids = get_top_k_documents(query, bm25, doc_ids, k=10)
        result[qid].extend(top_k_doc_ids)

        qcnt += 1
        if qcnt % 10 == 0:
            print(f"qcnt: {qcnt}")

    rankings_path = f"/home/work/retrieval/data/rankings/bm25.txt"
    write_file(rankings_path, result)
    eval_log_path = f"/home/work/retrieval/data/evals/bm25.txt"
    evaluate_dataset(query_path, rankings_path, eval_doc_count, eval_log_path)


if __name__ == "__main__":
    for i in range(9):
        do_expermient(i)
