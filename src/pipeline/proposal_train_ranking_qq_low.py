import pickle
import random
import time
from typing import List
from collections import defaultdict
import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from clusters import (
    Stream,
    Cluster,
    RandomProjectionLSH,
    DiversityBufferManager,
    assign_instance_or_add_cluster,
    clear_invalid_clusters,
    evict_clusters,
    initialize,
    renew_data,
    clear_unused_documents,
    build_cluster_cache_table_by_cosine,
    get_samples_top_and_farthest3_with_cache,
    get_samples_top_bottom_3_with_cache,
    visualize_clusters,
)
from data import read_jsonl, read_jsonl_as_dict, write_file, write_line
from functions import (
    InfoNCELoss,
    evaluate_dataset,
    get_top_k_documents,
    get_top_k_documents_partitioned,
)

torch.autograd.set_detect_anomaly(True)
tokenizer = BertTokenizer.from_pretrained("/home/work/.default/huijeong/bert_local")

num_gpus = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]


from multiprocessing import Pool, cpu_count
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


def merge_nested_dicts(d1, d2):
    result = defaultdict(lambda: defaultdict(list))
    for d in [d1, d2]:
        for cluster_id, qid_dict in d.items():
            for qid, lengths in qid_dict.items():
                result[cluster_id][qid].extend(lengths)
    # defaultdict -> dict 변환
    return {k: dict(v) for k, v in result.items()}


def build_query_caches(clusters, docs, query_result):
    all_qids = []
    for cluster in clusters:
        all_qids.extend(cluster.get_only_qids(docs))
    q_caches = build_cluster_cache_table_by_cosine(
        qids=all_qids,
        clusters=clusters,
        docs=docs,
        # query_result=query_result,
        query_batch_size=64,
        doc_batch_size=512,
    )
    return all_qids, q_caches


def add_positive_caches(all_qids, q_caches, clusters, docs):
    all_pids = []
    for qid in all_qids:
        pids, _ = get_samples_top_bottom_3_with_cache(
            caches=q_caches,
            query=docs[qid],
            docs=docs,
            clusters=clusters,
            positive_k=1,
            negative_k=1,
            ts=None,
            use_tensor_key=True,
        )
        all_pids.extend(pids)
    print(
        f"build_positive_caches | all_qids: {len(all_qids)}, all_pids: {len(all_pids)}"
    )
    p_caches = build_cluster_cache_table_by_cosine(
        qids=all_pids,
        clusters=clusters,
        docs=docs,
        query_batch_size=64,
        doc_batch_size=512,
    )
    total_ids = all_qids + all_pids
    final_caches = merge_nested_dicts(q_caches, p_caches)
    return total_ids, final_caches


def streaming_train(
    caches,
    queries,
    docs,
    ts,
    clusters: List[Cluster],
    model,
    lsh: RandomProjectionLSH,
    num_epochs,
    session_number,
    positive_k=1,
    negative_k=6,
    learning_rate=2e-5,
    batch_size=32,
    use_label=False,
    use_weight=False,
    use_tensor_key=False,
):
    query_cnt = len(queries)
    loss_fn = InfoNCELoss()
    learning_rate = learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    time_values = []
    keep_doc_ids = set()
    for epoch in range(num_epochs):
        total_loss, total_sec, batch_cnt = 0, 0, 0

        start_time = time.time()
        for start_idx in range(0, query_cnt, batch_size):
            end_idx = min(start_idx + batch_size, query_cnt)
            print(f"query {start_idx}-{end_idx}")
            query_batch = []
            pos_docs_all, neg_docs_all = [], []  # flat list of all pos/neg texts
            for idx in range(start_idx, end_idx):
                query = queries[idx]
                pos_ids, neg_ids = get_samples_top_and_farthest3_with_cache(
                    caches=caches,
                    query=query,
                    docs=docs,
                    clusters=clusters,
                    positive_k=positive_k,
                    negative_k=negative_k,
                    ts=ts,
                    use_tensor_key=use_tensor_key,
                )
                if idx == 0:
                    visualize_clusters(
                        clusters=clusters,
                        docs=docs,
                        save_path=f"../plot_{session_number}.png",
                        session_number=session_number,
                        representive_query_id=query["doc_id"],
                        representive_doc_ids=pos_ids + neg_ids,
                    )
                keep_doc_ids.add(query["doc_id"])
                keep_doc_ids.update(pos_ids)
                keep_doc_ids.update(neg_ids)
                pos_docs = [docs[_id]["text"] for _id in pos_ids]
                neg_docs = [docs[_id]["text"] for _id in neg_ids]
                query_batch.append(query["text"])
                pos_docs_all.extend(pos_docs)  # flatten
                neg_docs_all.extend(neg_docs)  # flatten
            # batch_size 개 쿼리 → (batch_size, embedding_dim)
            query_embeddings = encode_texts(model=model, texts=query_batch)
            # 전체 긍정/부정 텍스트 → (batch_size * K, embedding_dim)
            pos_embeddings = encode_texts(model=model, texts=pos_docs_all)
            neg_embeddings = encode_texts(model=model, texts=neg_docs_all)
            # → (batch_size, K, embedding_dim)
            positive_embeddings = pos_embeddings.view(
                -1, positive_k, pos_embeddings.shape[-1]
            )
            negative_embeddings = neg_embeddings.view(
                -1, negative_k, neg_embeddings.shape[-1]
            )

            loss = loss_fn(query_embeddings, positive_embeddings, negative_embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            time_values.append(loss.item())  # loss.item()
            batch_cnt += 1
            print(
                f"Processed {end_idx}/{query_cnt} queries | Batch Loss: {loss.item():.4f} | Total Loss: {total_loss / batch_cnt:.4f}"
            )

        end_time = time.time()
        execution_time = end_time - start_time
        total_sec += execution_time
        print(
            f"Epoch {epoch} | Total {total_sec} seconds, Avg {total_sec / batch_cnt} seconds."
        )
    return time_values, ts, keep_doc_ids


def train(
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
    light_weight=False,
    light_weight_rate=0.0,
):
    total_time_values = []
    required_doc_size = (
        required_doc_size if required_doc_size is not None else positive_k + negative_k
    )

    random_vectors = torch.randn(nbits, 768)
    lsh = RandomProjectionLSH(
        random_vectors=random_vectors, embedding_dim=768, use_tensor_key=use_tensor_key
    )
    prev_docs, clusters, query_result = None, [], {}
    diversity_buffer_manager = DiversityBufferManager()

    for session_number in range(start_session_number, end_session_number):
        ts = session_number
        time_values_path = (
            f"../data/loss/total_time_values_datasetM_large_share_30_{session_number}.txt"
        )
        print(f"Training Session {session_number}/{load_cluster}")
        start_time = time.time()
        stream = Stream(
            session_number=session_number,
            query_path=f"../data/datasetM_large_share/train_session{session_number}_queries.jsonl",
            doc_path=f"../data/datasetM_large_share/train_session{session_number}_docs.jsonl",
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
        if session_number != 0:
            print("Load last session model.")
            model_path = f"../data/model/datasetM_large_share{session_number-1}.pth"
            model.load_state_dict(torch.load(model_path, map_location=devices[-1]))
        else:
            print("Load Warming up model.")
            # model_path = f"../data/base_model_lotte.pth"
        model.train()
        new_model_path = f"../data/model/datasetM_large_share{session_number}.pth"

        # Initial : 매번 로드 or 첫 세션만 로드
        if (
            (load_cluster)
            or (not load_cluster and session_number == start_session_number)
            and session_number > 0
        ):
            with open(
                f"../data/clusters_datasetM_large_share_30_{session_number-1}.pkl",
                "rb",
            ) as f:
                print(f"Load last clusters.")
                clusters = pickle.load(f)
            with open(
                f"../data/prev_docs_datasetM_large_share_30_{session_number-1}.pkl",
                "rb",
            ) as f:
                print(f"Load last docs.")
                prev_docs = pickle.load(f)
                stream.docs.update(prev_docs)
            with open(
                f"../data/random_vectors_datasetM_large_share_30_{session_number-1}.pkl",
                "rb",
            ) as f:
                print(f"Load last random vectors.")
                random_vectors = pickle.load(f)
            # with open(
            #     f"../data/query_result_datasetM_large_share_30_hash9_{session_number-1}.pkl", "rb"
            # ) as f:
            #     print(f"Load last query_result.")
            #     last_query_result = pickle.load(f)
            #     query_result.update(last_query_result)
            # with open(
            #     f"../data/diversity_buffer_manager_datasetM_large_share_30_hash9_{session_number-1}.pkl",
            #     "rb",
            # ) as f:
            #     diversity_buffer_manager = pickle.load(f)
            batch_start = 0
        else:
            if session_number == 0:
                start_time = time.time()
                if warming_up_method == "initial_cluster":
                    init_k = (
                        int(np.log2(len(stream.initial_docs)))
                        if init_k is None
                        else init_k
                    )
                    clusters = initialize(
                        stream,
                        stream.initial_docs,
                        stream.docs,
                        init_k,
                        nbits,
                        lsh,
                        max_iters,
                        use_tensor_key,
                    )
                    initial_size = len(stream.initial_docs)
                    batch_start = 0
                elif warming_up_method == "query_seed":
                    init_k = (
                        int(np.log2(len(stream.stream_queries[0])))
                        if init_k is None
                        else init_k
                    )
                    clusters = initialize(
                        stream.stream_queries[0],
                        stream.docs,
                        init_k,
                        nbits,
                        max_iters,
                        use_tensor_key,
                    )
                    initial_size = len(stream.stream_queries[0])
                    batch_start = 1
                elif warming_up_method == "stream_seed":
                    init_k = (
                        int(np.log2(len(stream.stream_docs[0])))
                        if init_k is None
                        else init_k
                    )
                    clusters = initialize(
                        stream.stream_docs[0],
                        stream.docs,
                        init_k,
                        nbits,
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
        total_assign_time = 0
        for i in range(batch_start, len(stream.stream_docs)):
            print(f"Assign {i}th stream starts.")
            start_time = time.time()
            assign_instance_or_add_cluster(
                lsh=lsh,
                clusters=clusters,
                cluster_min_size=cluster_min_size,
                stream_docs=stream.stream_docs[i],
                docs=stream.docs,
                ts=ts,
                use_tensor_key=use_tensor_key,
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

        # Remain only trainable clusters
        clusters = clear_invalid_clusters(clusters, stream.docs, required_doc_size)

        print(
            f"=================================================BUILD QUERY CACHES======================================================"
        )
        start_time = time.time()
        all_qids, q_caches = build_query_caches(clusters, stream.docs, query_result)
        end_time = time.time()
        print(
            f"==============================================DONE({end_time-start_time}sec)===================================================="
        )
        write_line(
            time_values_path, f"BuildQueryCaches({end_time-start_time}sec)\n", "a"
        )
        # Train
        start_time = time.time()
        train_queries = diversity_buffer_manager.get_samples(
            docs=stream.docs,
            clusters=clusters,
            caches=q_caches,
            sample_size=len(stream.queries),
        )
        end_time = time.time()

        print(
            f"############################################QuerySelection({end_time - start_time}sec)############################################"
        )
        write_line(time_values_path, f"QuerySelection({end_time-start_time}sec)\n", "a")

        print(
            f"=================================================BUILD POSITIVE CACHES======================================================"
        )
        start_time = time.time()
        all_qids = [q["doc_id"] for q in train_queries]
        total_ids, cluster_caches = add_positive_caches(
            all_qids, q_caches, clusters, stream.docs
        )
        end_time = time.time()
        print(
            f"==============================================DONE({end_time-start_time}sec)===================================================="
        )
        write_line(
            time_values_path, f"BuildPositiveCaches({end_time-start_time}sec)\n", "a"
        )

        start_time = time.time()
        time_values, ts, keep_doc_ids = streaming_train(
            caches=cluster_caches,
            queries=train_queries,
            docs=stream.docs,
            ts=ts,
            clusters=clusters,
            model=model,
            lsh=lsh,
            num_epochs=num_epochs,
            negative_k=negative_k,
            batch_size=batch_size,
            use_label=use_label,
            use_weight=use_weight,
            session_number=session_number,
        )
        end_time = time.time()

        print(
            f"############################################Train({end_time - start_time}sec)############################################"
        )
        write_line(time_values_path, f"Train({end_time-start_time}sec)\n", "a")
        torch.save(model.state_dict(), new_model_path)
        # Evaluate
        # clusters, eval_stream_docs = evaluate_with_cluster(
        #     session_number=session_number,
        #     ts=ts,
        #     random_vectors=random_vectors,
        #     clusters=clusters,
        #     model_path=new_model_path,
        #     use_tensor_key=use_tensor_key,
        # )
        # _evaluate(session_number)

        # Evict
        start_time = time.time()
        evict_clusters(
            model,
            lsh,
            stream.docs,
            clusters,
            ts,
            required_doc_size,
            light_weight,
            light_weight_rate,
            keep_doc_ids,
        )
        # visualize_clusters(clusters, stream.docs, f"../cluster_plot_{session_number}_right_after_eviction.png")
        stream.docs = clear_unused_documents(clusters, stream.docs)
        # Accumulate
        prev_docs = stream.docs  # {**stream.docs, **eval_stream_docs}
        end_time = time.time()
        print(
            f"############################################Eviction({end_time - start_time}sec)############################################"
        )
        write_line(time_values_path, f"Eviction({end_time-start_time}sec)\n", "a")

        with open(
            f"../data/clusters_datasetM_large_share_30_{session_number}.pkl", "wb"
        ) as f:
            pickle.dump(clusters, f)
        with open(
            f"../data/prev_docs_datasetM_large_share_30_{session_number}.pkl", "wb"
        ) as f:
            pickle.dump(prev_docs, f)
        with open(
            f"../data/random_vectors_datasetM_large_share_30_{session_number}.pkl",
            "wb",
        ) as f:
            pickle.dump(random_vectors, f)
        # with open(f"../data/query_result_datasetM_large_share_30_hash9_{session_number}.pkl", "wb") as f:
        #     pickle.dump(query_result, f)
        # with open(
        #     f"../data/diversity_buffer_manager_datasetM_large_share_30_hash9_{session_number}.pkl", "wb"
        # ) as f:
        #     pickle.dump(diversity_buffer_manager, f)


def evaluate_with_cluster(
    session_number,
    ts,
    random_vectors,
    use_tensor_key,
    model_path,
    clusters: List[Cluster],
) -> List[Cluster]:
    eval_query_path = (
        f"../data/datasetM_large_share/test_session{session_number}_queries.jsonl"
    )
    eval_doc_path = (
        f"../data/datasetM_large_share/test_session{session_number}_docs.jsonl"
    )
    stream = Stream(
        session_number=session_number,
        query_path=eval_query_path,
        doc_path=eval_doc_path,
        warming_up_method="eval",
    )
    eval_query_count = len(stream.queries)
    eval_doc_count = len(stream.docs)
    print(
        f"Evaluate session {session_number} | #Query:{eval_query_count}, #Document:{eval_doc_count}"
    )
    # Assign and Retrieve
    # start_time = time.time()
    # model = BertModel.from_pretrained("bert-base-uncased").to(
    #     devices[-1]
    # )
    # model.load_state_dict(torch.load(model_path, weights_only=True))
    # model.eval()
    # result = retrieve_top_k_docs_from_cluster(
    #     model, stream, clusters, random_vectors, use_tensor_key, 10
    # )
    # end_time = time.time()
    # print(f"Spend {end_time-start_time} seconds for retrieval.")

    # rankings_path = f"../data/rankings/datasetM_large_share_hash9_{session_number}_with_cluster.txt"
    # write_file(rankings_path, result)
    # eval_log_path = f"../data/evals/proposa_datasetM_large_share_hash9_{session_number}_with_cluster.txt"
    # evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
    return clusters, stream.docs


def evaluate(session_count=10):
    for session_number in range(10):
        _evaluate(session_number, False)


def _evaluate(session_number, partition=False):
    method = "datasetM_large_share"
    print(f"Evaluate Session {session_number}")
    eval_query_path = (
        f"../data/datasetM_large_share/test_session{session_number}_queries.jsonl"
    )
    eval_doc_path = (
        f"../data/datasetM_large_share/train_session{session_number}_docs.jsonl"
    )
    # eval_doc_path = f"../data/datasetM_large_share/test_session{session_number}_docs.jsonl"

    eval_query_data = read_jsonl(eval_query_path, True)
    eval_doc_data = read_jsonl(eval_doc_path, False)

    eval_query_count = len(eval_query_data)
    eval_doc_count = len(eval_doc_data)
    print(f"Query count:{eval_query_count}, Document count:{eval_doc_count}")

    rankings_path = f"../data/rankings/{method}_session_{session_number}.txt"
    model_path = f"../data/model/{method}{session_number}.pth"
    # model_path = f"../data/model/{method}_session_{session_number}.pth"

    if not partition:
        start_time = time.time()
        new_q_data, new_d_data = renew_data(
            queries=eval_query_data,
            documents=eval_doc_data,
            model_path=model_path,
            nbits=12,
            renew_q=True,
            renew_d=True,
            use_tensor_key=True,
        )
        end_time = time.time()
        print(f"Spend {end_time-start_time} seconds for encoding.")

    if partition:
        start_time = time.time()
        result = get_top_k_documents_partitioned(eval_query_data, eval_doc_data)
        end_time = time.time()
    else:
        start_time = time.time()
        result = get_top_k_documents(new_q_data, new_d_data)
        end_time = time.time()
    print(f"Spend {end_time-start_time} seconds for retrieval.")

    rankings_path = f"../data/rankings/{method}_session_{session_number}.txt"
    write_file(rankings_path, result)
    eval_log_path = f"../data/evals/{method}_{session_number}.txt"
    evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
    # del new_q_data, new_d_data


def eval_rankings(session_number):
    eval_query_path = (
        f"../data/datasetM_large_share/test_session{session_number}_queries.jsonl"
    )
    eval_doc_path = (
        f"../data/datasetM_large_share/test_session{session_number}_docs.jsonl"
    )

    eval_query_data = read_jsonl(eval_query_path, True)
    eval_doc_data = read_jsonl(eval_doc_path, False)

    eval_query_count = len(eval_query_data)
    eval_doc_count = len(eval_doc_data)
    print(f"Query count:{eval_query_count}, Document count:{eval_doc_count}")

    rankings_path = (
        f"../data/rankings/datasetM_large_share_30_{session_number}_with_cluster.txt"
    )
    eval_log_path = (
        f"../data/evals/datasetM_large_share_30_{session_number}_with_cluster.txt"
    )
    evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
    rankings_path = f"../data/rankings/datasetM_large_share{session_number}.txt"
    evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
