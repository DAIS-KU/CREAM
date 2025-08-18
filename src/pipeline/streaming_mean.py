import pickle
import random
import time
from typing import List
from collections import defaultdict

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from ablation import (
    assign_instance_or_add_cluster,
    clear_invalid_clusters,
    evict_clusters,
    initialize,
    clear_unused_documents,
    Stream,
)


torch.autograd.set_detect_anomaly(True)
tokenizer = BertTokenizer.from_pretrained(
    "/home/work/retrieval/bert-base-uncased/bert-base-uncased"
)

num_gpus = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]


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

def train(
    start_session_number=0,
    end_sesison_number=12,
    load_cluster=True,
    sampling_rate=None,
    sampling_size_per_query=100,
    num_epochs=1,
    batch_size=32,
    warmingup_rate="none",
    positive_k=1,
    negative_k=6,
    cluster_min_size=50,
    nbits=12,
    max_iters=3,
    init_k=12,
    use_label=False,
    use_weight=False,
    use_tensor_key=False,
    warming_up_method="stream_seed",
    required_doc_size=20,
):
    required_doc_size = (
        required_doc_size if required_doc_size is not None else positive_k + negative_k
    )

    prev_docs, clusters = None, []

    for session_number in range(start_session_number, end_sesison_number):
        ts = session_number
        model = BertModel.from_pretrained(
            "/home/work/retrieval/bert-base-uncased/bert-base-uncased"
        ).to(devices[-1])
        model_path = None
        if session_number != 0:
            print("Load last session model.")
            model_path = (
                f"../data/model/proposal_wo_term_session_{session_number-1}.pth"
            )
            model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
        model.train()
        new_model_path = f"../data/model/proposal_wo_term_session_{session_number}.pth"

        print(f"Training Session {session_number}/{load_cluster}")
        stream = Stream(
            session_number=session_number,
            model_path=model_path,
            query_path=f"/home/work/retrieval/data/datasetL_large_share/train_session{session_number}_queries.jsonl",
            doc_path=f"/home/work/retrieval/data/datasetL_large_share/train_session{session_number}_docs.jsonl",
            warmingup_rate=warmingup_rate,
            sampling_rate=sampling_rate,
            prev_docs=prev_docs,
            sampling_size_per_query=sampling_size_per_query,
            warming_up_method=warming_up_method,
        )
        print(f"Session {session_number} | Document count:{len(stream.docs.keys())}")
        # Initial : 매번 로드 or 첫 세션만 로드
        if (load_cluster or session_number == start_session_number) and (
            session_number > 0
        ):
            print(f"Load last sesion clusters, docs.")
            # with open(f"../data/clusters_wo_term_{session_number-1}.pkl", "rb") as f:
            #     clusters = pickle.load(f)
            #     print("Cluster loaded.")
            # with open(f"../data/prev_docs_wo_term_{session_number-1}.pkl", "rb") as f:
            #     prev_docs = pickle.load(f)
            #     print("Prev_docs loaded.")
            # stream.docs.update(prev_docs)
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
                    clusters = initialize(
                        model,
                        stream.stream_docs[0],
                        stream.docs,
                        init_k,
                        max_iters,
                        use_tensor_key,
                    )
                    initial_size = len(stream.stream_docs[0])
                    batch_start = 1
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
        for i in range(batch_start, len(stream.stream_docs)):
            print(f"Assign {i}th stream starts.")
            start_time = time.time()
            assign_instance_or_add_cluster(
                model=model,
                clusters=clusters,
                cluster_min_size=cluster_min_size,
                stream_docs=stream.stream_docs[i],
                docs=stream.docs,
                ts=ts,
                use_tensor_key=use_tensor_key,
            )
            # if i % 50 == 0:
            #     for j, cluster in enumerate(clusters):
            #         print(f"{j}th size: {len(cluster.doc_ids)}")
            end_time = time.time()
            print(f"Assign {i}th stream ended({end_time - start_time}sec).")

        # Remain only trainable clusters
        clusters = clear_invalid_clusters(clusters, stream.docs, required_doc_size)

        # Evict
        clusters = evict_clusters(model, stream.docs, clusters, ts, required_doc_size)
        stream.docs = clear_unused_documents(clusters, stream.docs)
        # Accumulate **eval_stream_docs
        prev_docs = stream.docs

        # with open(f"../data/clusters_wo_term_{session_number}.pkl", "wb") as f:
        #     pickle.dump(clusters, f)
        #     print("Cluster dumped.")
        # with open(f"../data/prev_docs_wo_term_{session_number}.pkl", "wb") as f:
        #     pickle.dump(prev_docs, f)
        #     print("Prev_docs dumped.")
