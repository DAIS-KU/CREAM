from typing import List, Dict
from ablation import (
    Cluster,
    encode_cluster_data_mean_pooling,
    kmeans_mean_pooling,
    find_k_closest_clusters,
    assign_instance_or_add_cluster_doc2cluster,
)
from functions import renew_data_mean_pooling
from data import read_jsonl, read_jsonl_as_dict
from transformers import BertModel
import torch

num_gpus = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]


def initialize(model, docs, k, max_iters, use_tensor_key=True) -> List[Cluster]:
    enoded_stream_docs = encode_cluster_data_mean_pooling(
        documents_data=list(docs.values()),
        model=model,
    )
    enoded_stream_docs = list(enoded_stream_docs.values())
    centroids, cluster_instances = kmeans_mean_pooling(enoded_stream_docs, k, max_iters)

    clusters, doc2cluster = [], {}
    for cid, centroid in enumerate(centroids):
        if len(cluster_instances[cid]):
            print(f"Create {len(clusters)}th Cluster.")
            clusters.append(
                Cluster(model, centroid, cluster_instances[cid], docs)
            )
            for doc in cluster_instances[cid]:
                doc2cluster[doc["doc_id"]] = cid
    return clusters, doc2cluster


def model_builder(model_path=None):
    return BertModel.from_pretrained("/home/work/.default/huijeong/bert_local")


def evaluate_success_recall(queries, clusters, doc2cluster, verbose=False):
    cluster_docs = {}
    for doc_id, c_id in doc2cluster.items():
        cluster_docs.setdefault(c_id, set()).add(doc_id)

    total_hits = 0
    total_rels = 0
    success_cnt = 0
    recall_vals = []

    for q in queries:
        q_emb = q["EMB"]
        ans_pids = q["answer_pids"]
        closest = find_k_closest_clusters(
            model=None,
            embs=q_emb.unsqueeze(0),
            clusters=clusters,
            k=1,
            device=devices[-1],
        )[0][0]
        docs_in_c = cluster_docs[closest]
        # print(f"Docs in cluster {closest}: {docs_in_c}")

        hits = sum(1 for pid in ans_pids if pid in docs_in_c)
        rels = len(ans_pids)  # > 0 (가정)
        recall = hits / rels
        success = 1 if hits > 0 else 0

        total_hits += hits
        total_rels += rels
        success_cnt += success
        recall_vals.append(recall)
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
    return summary


def streaming_assign(clusters, doc2cluster, _query_dict, _doc_dict, dataset):
    query_path = (
        f"/home/work/.default/huijeong/data/pooled_2_{dataset}/pooled_queries.jsonl"
    )
    doc_path = f"/home/work/.default/huijeong/data/pooled_2_{dataset}/pooled_docs.jsonl"
    queries = read_jsonl(query_path, True)
    docs = read_jsonl(doc_path, False)

    query_dict, doc_dict = renew_data_mean_pooling(model_builder, None, queries, docs)
    total_docs = {**query_dict, **doc_dict, **_query_dict, **_doc_dict}
    stream_docs = {**query_dict, **doc_dict}
    # print(f"total docs: {len(total_docs)}, stream docs: {len(stream_docs)}")
    model = model_builder()
    clusters, doc2cluster = assign_instance_or_add_cluster_doc2cluster(
        model=model,
        clusters=clusters,
        stream_docs=list(stream_docs.values()),
        docs=total_docs,
        doc2cluster=doc2cluster,
    )
    for q in queries:
        query_dict[q["qid"]]["answer_pids"] = q["answer_pids"]
    return clusters, doc2cluster, query_dict, doc_dict


def static_assign(
    dataset,
):
    query_path = (
        f"/home/work/.default/huijeong/data/pooled_1_{dataset}/pooled_queries.jsonl"
    )
    doc_path = f"/home/work/.default/huijeong/data/pooled_1_{dataset}/pooled_docs.jsonl"
    queries = read_jsonl(query_path, True)
    docs = read_jsonl(doc_path, False)

    query_dict, doc_dict = renew_data_mean_pooling(model_builder, None, queries, docs)
    for q in queries:
        query_dict[q["qid"]]["answer_pids"] = q["answer_pids"]
    model = model_builder()
    clusters, doc2cluster = initialize(model, doc_dict, 5, 5, False)
    return clusters, doc2cluster, queries, query_dict, doc_dict


def streaming_mean_evaluation(dataset="lotte"):
    clusters, doc2cluster, _, query_dict1, doc_dict1 = static_assign(dataset)
    print("Initialization complete.")
    clusters, doc2cluster, query_dict2, _ = streaming_assign(
        clusters, doc2cluster, query_dict1, doc_dict1, dataset
    )
    print("Streaming assignment complete.")
    summary, _ = evaluate_success_recall(
        {**query_dict1, **query_dict2}, clusters, doc2cluster, verbose=False
    )
    print("Evaluation complete.")
    print(summary)
