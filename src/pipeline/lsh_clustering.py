from typing import List
from clusters import Cluster, kmeans_pp_use_tensor_key, find_k_closest_clusters
from clusters import renew_data
from data import read_jsonl, read_jsonl_as_dict
from transformers import BertModel
import torch

num_gpus = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]


def initialize(docs, k, nbits, max_iters, use_tensor_key=True) -> List[Cluster]:
    _, encoded_docs = renew_data(
        queries=None,
        documents=list(docs.values()),
        renew_q=False,
        renew_d=True,
        nbits=nbits,
        use_tensor_key=use_tensor_key,
        model_path=None,
    )
    centroids, cluster_instances = kmeans_pp_use_tensor_key(
        list(encoded_docs.values()), k, max_iters, nbits
    )
    clusters, doc2cluster = [], {}
    for cid, centroid in enumerate(centroids):
        if len(cluster_instances[cid]):
            print(f"Create {len(clusters)}th Cluster.")
            clusters.append(
                Cluster(centroid, cluster_instances[cid], encoded_docs, use_tensor_key)
            )
            for doc in cluster_instances[cid]:
                doc2cluster[doc["doc_id"]] = cid
    return clusters, doc2cluster


def evaluate_success_recall(queries, clusters, doc2cluster, verbose=False):
    cluster_docs = {}
    for doc_id, c_id in doc2cluster.items():
        cluster_docs.setdefault(c_id, set()).add(doc_id)
    per_query = []
    total_hits = 0
    total_rels = 0
    success_cnt = 0
    recall_vals = []

    for q in list(queries.values()):
        q_emb = q["TOKEN_EMBS"]
        ans_pids = q["answer_pids"]
        closest = find_k_closest_clusters(
            token_embs=[q_emb],
            clusters=clusters,
            k=1,
            device=devices[-1],
            use_tensor_key=True,
        )[0][0]
        docs_in_c = cluster_docs[closest]

        hits = sum(1 for pid in ans_pids if pid in docs_in_c)
        rels = len(ans_pids)  # > 0 (가정)
        recall = hits / rels
        success = 1 if hits > 0 else 0

        total_hits += hits
        total_rels += rels
        success_cnt += success
        recall_vals.append(recall)

        per_query.append(
            {
                "text": q.get("text", ""),
                "assigned_cluster": c_id,
                "rel_in_cluster": hits,
                "rel_total": rels,
                "recall": recall,
                "success": success,
            }
        )

        if verbose:
            print(
                f"Query: {q.get('text','')}\n  Cluster={c_id}  hits={hits}/{rels}  recall={recall:.3f}  success={success}"
            )

    macro_recall = sum(recall_vals) / len(recall_vals) if recall_vals else 0.0
    micro_recall = total_hits / total_rels if total_rels > 0 else 0.0
    success_rate = success_cnt / len(recall_vals) if recall_vals else 0.0

    summary = {
        "num_evaluated_queries": len(recall_vals),
        "macro_recall": macro_recall,
        "micro_recall": micro_recall,
        "success@1_rate": success_rate,
    }
    return summary, per_query


def static_assign_evaluation(nbits=3):
    query_path = "/home/work/.default/huijeong/data/pooled/pooled_queries.jsonl"
    doc_path = "/home/work/.default/huijeong/data/pooled/pooled_docs.jsonl"
    queries = read_jsonl(query_path, True)
    docs = read_jsonl_as_dict(doc_path, "doc_id")

    query_dict, _ = renew_data(
        queries=queries,
        documents=None,
        renew_q=True,
        renew_d=False,
        nbits=nbits,
        use_tensor_key=True,
        model_path=None,
    )
    clusters, doc2cluster = initialize(docs, 5, nbits, 5, use_tensor_key=True)
    for q in queries:
        query_dict[q["qid"]]["answer_pids"] = q["answer_pids"]
    summary, per_query = evaluate_success_recall(
        queries=query_dict,
        clusters=clusters,
        doc2cluster=doc2cluster,
        verbose=False,
    )

    print("=== Success/Recall Summary ===")
    print(f"#eval queries : {summary['num_evaluated_queries']}")
    print(
        f"macro recall  : {summary['macro_recall']:.4f}"
        if summary["macro_recall"] is not None
        else "macro recall  : N/A"
    )
    print(
        f"micro recall  : {summary['micro_recall']:.4f}"
        if summary["micro_recall"] is not None
        else "micro recall  : N/A"
    )
    print(
        f"success@1 rate: {summary['success@1_rate']:.4f}"
        if summary["success@1_rate"] is not None
        else "success@1 rate: N/A"
    )
