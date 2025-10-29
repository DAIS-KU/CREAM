import pickle
import random
import time
from typing import List
from collections import defaultdict
import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from clusters import (
    Cluster,
    Stream,
    RandomProjectionLSH,
    assign_instance_or_add_cluster_doc2cluster,
    clear_invalid_clusters,
    evict_clusters,
    initialize_doc2cluster,
    clear_unused_documents,
    find_k_closest_clusters,
)
from data import write_line

torch.autograd.set_detect_anomaly(True)
tokenizer = BertTokenizer.from_pretrained("/home/work/.default/huijeong/bert_local")

num_gpus = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]


import torch


def encode_texts(model, texts, max_length=256):
    device = model.device
    no_padding_inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    no_padding_inputs = {
        key: value.to(device) for key, value in no_padding_inputs.items()
    }
    outputs = model(**no_padding_inputs).last_hidden_state
    embedding = outputs[:, 0, :]  # [CLS]만 사용
    return embedding


def evaluate_success_recall(queries, clusters, doc2cluster, verbose=False):
    cluster_docs = {}
    for doc_id, c_id in doc2cluster.items():
        cluster_docs.setdefault(c_id, set()).add(doc_id)
    total_hits = 0
    total_rels = 0
    success_cnt = 0
    recall_vals = []

    for q in queries:
        q_emb = q["TOKEN_EMBS"]
        ans_pids = q["answer_pids"]
        closest = find_k_closest_clusters(
            token_embs=[q_emb],
            clusters=clusters,
            k=3,
            device=devices[-1],
            use_tensor_key=True,
        )[0]
        docs_in_c = set()
        docs_in_c.update(cluster_docs[closest[0]])
        docs_in_c.update(cluster_docs[closest[1]])
        docs_in_c.update(cluster_docs[closest[2]])

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


def get_sse(docs, clusters: List[Cluster]):
    SSE = 0.0
    for cluster in clusters:
        SSE += cluster.get_sse(docs)
    return SSE


def streaming_lsh_evaluation(
    start_session_number=0,
    end_session_number=10,
    load_cluster=False,
    sampling_rate=None,
    sampling_size_per_query=30,
    num_epochs=1,
    batch_size=32,
    warmingup_rate=0.2,
    positive_k=1,
    negative_k=6,
    cluster_min_size=10,
    nbits=12,
    max_iters=3,
    init_k=None,
    use_label=False,
    use_weight=False,
    use_tensor_key=False,
    warming_up_method=None,
    required_doc_size=None,
    include_answer=False,
):
    required_doc_size = (
        required_doc_size if required_doc_size is not None else positive_k + negative_k
    )

    random_vectors = torch.randn(nbits, 768)
    lsh = RandomProjectionLSH(
        random_vectors=random_vectors, embedding_dim=768, use_tensor_key=use_tensor_key
    )
    prev_docs, clusters, doc2cluster = None, [], {}
    sse_path = f"/home/work/.default/huijeong/data/sse/nbits_{nbits}_lotte.txt"

    for session_number in range(start_session_number, end_session_number):
        ts = session_number
        time_values_path = f"/home/work/.default/huijeong/data/loss/total_time_values_proposal_lotte_{session_number}.txt"
        print(f"Training Session {session_number}/{load_cluster}")
        start_time = time.time()
        stream = Stream(
            session_number=session_number,
            query_path=f"/home/work/.default/huijeong/data/lotte_session/train_session{session_number}_queries.jsonl",
            doc_path=f"/home/work/.default/huijeong/data/lotte_session/train_session{session_number}_docs_filtered.jsonl",
            warmingup_rate=warmingup_rate,
            sampling_rate=sampling_rate,
            prev_docs=prev_docs,
            sampling_size_per_query=sampling_size_per_query,
            warming_up_method=warming_up_method,
            include_answer=include_answer,
        )
        # query_result.update(stream.query_result)
        print(f"Session {session_number} | Document count:{len(stream.docs.keys())}")
        end_time = time.time()
        print(
            f"############################################Initialize({end_time-start_time}sec)############################################"
        )
        write_line(time_values_path, f"Initialize({end_time-start_time}sec)\n", "a")

        model = BertModel.from_pretrained("/home/work/.default/huijeong/bert_local").to(
            devices[-1]
        )
        model.eval()

        # Initial : 매번 로드 or 첫 세션만 로드
        if (
            (load_cluster)
            or (not load_cluster and session_number == start_session_number)
            and session_number > 0
        ):
            # with open(
            #     f"/home/work/.default/huijeong/data/clusters_lotte_{session_number-1}.pkl",
            #     "rb",
            # ) as f:
            #     print(f"Load last clusters.")
            #     clusters = pickle.load(f)
            # with open(
            #     f"/home/work/.default/huijeong/data/prev_docs_lotte_{session_number-1}.pkl",
            #     "rb",
            # ) as f:
            #     print(f"Load last docs.")
            #     prev_docs = pickle.load(f)
            #     stream.docs.update(prev_docs)
            # with open(
            #     f"/home/work/.default/huijeong/data/random_vectors_lotte_{session_number-1}.pkl",
            #     "rb",
            # ) as f:
            #     print(f"Load last random vectors.")
            #     random_vectors = pickle.load(f)
            batch_start = 0
        else:
            if session_number == 0:
                start_time = time.time()
                if warming_up_method == "stream_seed":
                    init_k = (
                        int(np.log2(len(stream.stream_docs[0])))
                        if init_k is None
                        else init_k
                    )
                    clusters, doc2cluster, random_vectors = initialize_doc2cluster(
                        stream.stream_docs[0],
                        stream.docs,
                        init_k,
                        nbits,
                        max_iters,
                        use_tensor_key,
                    )
                    initial_size = len(stream.stream_docs[0])
                    batch_start = 1
                    lsh = RandomProjectionLSH(
                        random_vectors=random_vectors,
                        embedding_dim=768,
                        use_tensor_key=use_tensor_key,
                    )
                else:
                    raise NotImplementedError(
                        f"Unsupported warming_up_method: {warming_up_method}"
                    )
                end_time = time.time()
                print(
                    f"Spend {end_time-start_time} seconds for clustering({len(clusters)}, {initial_size}) warming up."
                )
            else:
                batch_start = 0
        # Assign stream batch
        total_assign_time = 0
        for i in range(batch_start, len(stream.stream_docs)):
            print(f"Assign {i}th stream starts.")
            start_time = time.time()
            clusters, doc2cluster = assign_instance_or_add_cluster_doc2cluster(
                lsh=lsh,
                clusters=clusters,
                stream_docs=stream.stream_docs[i],
                docs=stream.docs,
                ts=None,
                use_tensor_key=True,
                doc2cluster=doc2cluster,
            )
            if i % 50 == 0:
                for j, cluster in enumerate(clusters):
                    print(f"{j}th size: {len(cluster.doc_ids)}")
            end_time = time.time()
            total_assign_time += end_time - start_time
            print(f"Assign {i}th stream ended({end_time - start_time}sec).")
        print(
            f"############################################Assign({total_assign_time}sec)############################################"
        )
        write_line(time_values_path, f"Assign({total_assign_time}sec)\n", "a")

        # summary = evaluate_success_recall(stream.queries, clusters, doc2cluster, verbose=False)
        # print(f"Evaluation Summary: {summary}")
        summary = get_sse(stream.docs, clusters)
        print(f"Session {session_number} | SSE: {summary}")
        write_line(sse_path, f"Session {session_number} | SSE: {summary}\n", "a")

        # Remain only trainable clusters
        clusters = clear_invalid_clusters(clusters, stream.docs, required_doc_size)

        start_time = time.time()
        # Evict
        evict_clusters(model, lsh, stream.docs, clusters, ts, required_doc_size)
        stream.docs = clear_unused_documents(clusters, stream.docs)
        # Accumulate
        prev_docs = stream.docs  # {**stream.docs, **eval_stream_docs}
        end_time = time.time()
        print(
            f"############################################Eviction({end_time - start_time}sec)############################################"
        )
        write_line(time_values_path, f"Eviction({end_time-start_time}sec)\n", "a")

        # with open(
        #     f"/home/work/.default/huijeong/data/clusters_lotte_{session_number}.pkl", "wb"
        # ) as f:
        #     pickle.dump(clusters, f)
        # with open(
        #     f"/home/work/.default/huijeong/data/prev_docs_lotte_{session_number}.pkl", "wb"
        # ) as f:
        #     pickle.dump(prev_docs, f)
        # with open(
        #     f"/home/work/.default/huijeong/data/random_vectors_lotte_{session_number}.pkl",
        #     "wb",
        # ) as f:
        #     pickle.dump(random_vectors, f)
