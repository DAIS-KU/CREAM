from typing import List
from clusters import (
    Cluster,
    RandomProjectionLSH,
    kmeans_pp_use_tensor_key_random_vectors,
    find_k_closest_clusters,
    assign_instance_or_add_cluster_doc2cluster,
)
from clusters import renew_data
from data import read_jsonl, read_jsonl_as_dict
from transformers import BertModel
import torch

num_gpus = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]


def initialize(docs, k, nbits, max_iters, use_tensor_key=True):
    _, encoded_docs = renew_data(
        queries=None,
        documents=list(docs.values()),
        renew_q=False,
        renew_d=True,
        nbits=nbits,
        use_tensor_key=use_tensor_key,
        model_path=None,
    )
    centroids, cluster_instances, random_vectors = (
        kmeans_pp_use_tensor_key_random_vectors(
            list(encoded_docs.values()), k, max_iters, nbits
        )
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
    return clusters, doc2cluster, random_vectors


def streaming_assign(
    session_number,
    clusters,
    doc2cluster,
    random_vectors,
    _query_dict,
    _doc_dict,
    dataset="lotte",
    nbits=3,
):
    query_path = (
        f"/home/work/.default/huijeong/data/{dataset}_session/train_session{session_number}_queries.jsonl"
    )
    doc_path = f"/home/work/.default/huijeong/data/{dataset}_session/train_session{session_number}_docs.jsonl"
    queries = read_jsonl(query_path, True)
    docs = read_jsonl(doc_path, "doc_id")

    query_dict, doc_dict = renew_data(
        queries=queries,
        documents=docs,
        renew_q=True,
        renew_d=True,
        nbits=nbits,
        use_tensor_key=True,
        model_path=None,
    )
    # print(f"query_dict size:{len(query_dict)}, doc_dict size:{len(doc_dict)}, _query_dict:{len(_query_dict)}, _doc_dict:{len(_doc_dict)}")
    stream_docs = {**doc_dict, **query_dict}
    # print(f"stream_docs size: {len(stream_docs)}")
    total_docs = {**doc_dict, **query_dict, **_query_dict, **_doc_dict}
    # print(f"total_docs size: {len(total_docs)}")
    lsh = RandomProjectionLSH(
        random_vectors=random_vectors,
        embedding_dim=768,
        use_tensor_key=True,
    )
    clusters, doc2cluster = assign_instance_or_add_cluster_doc2cluster(
        lsh=lsh,
        clusters=clusters,
        stream_docs=list(stream_docs.values()),
        docs=total_docs,
        ts=None,
        use_tensor_key=True,
        doc2cluster=doc2cluster,
    )
    for q in queries:
        query_dict[q["qid"]]["answer_pids"] = q["answer_pids"]
    return doc2cluster, queries, query_dict, doc_dict


def static_assign(dataset="lotte", nbits=3):
    query_path = (
        f"/home/work/.default/huijeong/data/{dataset}_session/train_session0_queries.jsonl"
    )
    doc_path = f"/home/work/.default/huijeong/data/{dataset}_session/train_session0_docs.jsonl"
    queries = read_jsonl(query_path, True)
    docs = read_jsonl_as_dict(doc_path, "doc_id")

    query_dict, doc_dict = renew_data(
        queries=queries,
        documents=list(docs.values()),
        renew_q=True,
        renew_d=True,
        nbits=nbits,
        use_tensor_key=True,
        model_path=None,
    )
    clusters, doc2cluster, random_vectors = initialize(
        docs, 5, nbits, 5, use_tensor_key=True
    )
    for q in queries:
        query_dict[q["qid"]]["answer_pids"] = q["answer_pids"]
    return clusters, doc2cluster, random_vectors, queries, query_dict, doc_dict


def evaluate_success_recall(queries, clusters, doc2cluster, verbose=False):
    cluster_docs = {}
    for doc_id, c_id in doc2cluster.items():
        cluster_docs.setdefault(c_id, set()).add(doc_id)
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


def streaming_lsh_evaluation(dataset="lotte", nbits=3):
    clusters, doc2cluster, random_vectors, _, query_dict1, doc_dict1 = static_assign(
        dataset, nbits
    )
    print("Initialization complete.")
    doc2cluster, _, query_dict2, _ = streaming_assign(
        clusters, doc2cluster, random_vectors, query_dict1, doc_dict1, dataset, nbits
    )
    print("Streaming assignment complete.")
    summary = evaluate_success_recall(
        {**query_dict1, **query_dict2}, clusters, doc2cluster, verbose=False
    )
    print("Evaluation complete.")
    print(summary)
