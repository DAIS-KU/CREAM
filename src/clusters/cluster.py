import concurrent
import copy
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List

import bisect
import random
import torch

from functions import (
    calculate_S_qd_regl,
    calculate_S_qd_regl_batch,
    calculate_S_qd_regl_dict,
    get_passage_embeddings,
)

from .prototype import RandomProjectionLSH

MAX_SCORE = 256
num_devices = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_devices)]


class Cluster:
    def __init__(
        self,
        centroid,
        cluster_docs,
        docs: dict,
        use_tensor_key,
        timestamp=0,
        batch_size=64,
        z1=8.0,
        z2=0.25,  # lotte 0.25 msmarco 0.75
    ):
        self.prototype = centroid
        self.doc_ids = [d["doc_id"] for d in cluster_docs]
        self.use_tensor_key = use_tensor_key
        self.timestamp = timestamp
        self.batch_size = batch_size
        self.z1 = z1
        self.z2 = z2
        self.cache = {}
        if len(self.doc_ids) == 1:
            self.S1 = 0.0
            self.S2 = 0.0
            self.N = 1.0
        else:
            self.update_statistics(docs)
        # print(f"init | S1:{self.S1}, S2:{self.S2}, N:{self.N}, z1:{self.z1}, z2:{self.z2}")

    def get_only_docids(self, docs: dict, verbose=True):
        # self.doc_ids= self.doc_ids.intersection(docs.keys())
        only_doc_ids = [
            doc_id for doc_id in self.doc_ids if not docs[doc_id]["is_query"]
        ]
        if verbose:
            print(f"Document only #{len(only_doc_ids)}/{len(self.doc_ids)}")
        return only_doc_ids

    def get_only_qids(self, docs: dict):
        only_qids = [doc_id for doc_id in self.doc_ids if docs[doc_id]["is_query"]]
        print(f"Query only #{len(only_qids)}/{len(self.doc_ids)}")
        return only_qids

    # batch_ids 순서가 유지되는가?
    def get_docids_and_scores(self, query, docs: dict):
        query_token_embs = query["TOKEN_EMBS"]

        def process_batch(device, batch_doc_ids):
            regl_scores = []
            temp_query_token_embs = query_token_embs.clone().unsqueeze(0).to(device)
            for i in range(0, len(batch_doc_ids), self.batch_size):
                batch_ids = batch_doc_ids[i : i + self.batch_size]
                # batch_embs = torch.stack(
                #         [docs[doc_id]["TOKEN_EMBS"] for doc_id in batch_ids], dim=0
                #     )
                batch_key = tuple(batch_ids)
                # if batch_key in self.cache:
                #     print(f"[cache_HIT] Found cached embeddings for key({i}-{i+len(batch_ids)})")
                # else:
                #     print(f"[cache_MISS] Creating embeddings for key({i}-{i+len(batch_ids)})")
                batch_embs = self.cache.setdefault(
                    batch_key,
                    torch.stack(
                        [docs[doc_id]["TOKEN_EMBS"] for doc_id in batch_ids], dim=0
                    ),
                )
                if batch_embs.dim() > 3:
                    batch_embs = batch_embs.squeeze()
                # print(f"get_topk_docids({device}) | batch_embs:{batch_embs.shape}, temp_query_token_embs:{temp_query_token_embs.shape}")
                regl_score = calculate_S_qd_regl_batch(
                    temp_query_token_embs, batch_embs, device
                )
                regl_scores.append(regl_score)
            regl_scores = torch.cat(regl_scores, dim=0)
            # print(f"regl_scores: {regl_scores.shape}")
            return [
                (doc_id, regl_scores[idx].item())
                for idx, doc_id in enumerate(batch_doc_ids)
            ]

        regl_scores = []
        #  순서 보장 필요
        only_doc_ids = self.get_only_docids(docs)
        batch_cnt = min(num_devices, len(only_doc_ids))
        doc_ids_batches = [only_doc_ids[i::batch_cnt] for i in range(batch_cnt)]
        with ThreadPoolExecutor() as executor:
            futures = []
            for i, device in enumerate(range(batch_cnt)):
                futures.append(
                    executor.submit(process_batch, device, doc_ids_batches[i])
                )
            for future in futures:
                regl_scores.extend(future.result())

        combined_regl_scores = sorted(regl_scores, key=lambda x: x[1], reverse=True)
        return combined_regl_scores

    def get_topk_docids_and_scores(self, query, docs: dict, k):
        combined_regl_scores = self.get_docids_and_scores(query, docs)
        top_k_regl_docs = combined_regl_scores[:k]
        bottom_k_regl_docs = combined_regl_scores[-k:]
        return top_k_regl_docs, bottom_k_regl_docs  # [(id, score), ]

    def get_topk_docids_and_scores_with_cache(self, qid, cache: dict, docs: dict, k):
        doc_id_scores = cache[qid]
        top_k_docids_and_scores = doc_id_scores[:k]
        bottom_k_docids_and_scores = doc_id_scores[-k:]
        return top_k_docids_and_scores, bottom_k_docids_and_scores

    def get_topk_docids_with_cache(
        self, query, cache: dict, docs: dict, k
    ) -> List[str]:
        (
            top_k_regl_docs,
            bottom_k_regl_docs,
        ) = self.get_topk_bottomk_docids_and_scores_with_cache(
            qid=query["doc_id"], cache=cache, docs=docs, k=k
        )
        top_k_regl_doc_ids = [x[0] for x in top_k_regl_docs]
        bottom_k_regl_doc_ids = [x[0] for x in bottom_k_regl_docs]
        return top_k_regl_doc_ids, bottom_k_regl_doc_ids

    def get_topk_docids(self, query, docs: dict, k) -> List[str]:
        top_k_regl_docs, bottom_k_regl_docs = self.get_topk_docids_and_scores(
            query, docs, k
        )
        top_k_regl_doc_ids = [x[0] for x in top_k_regl_docs]
        bottom_k_regl_doc_ids = [x[0] for x in bottom_k_regl_docs]
        return top_k_regl_doc_ids, bottom_k_regl_doc_ids

    def calculate_mean(self):
        mean = self.S1 / self.N
        return mean

    def calculate_rms(self):
        mean_S1 = self.S1 / self.N
        mean_S2 = self.S2 / self.N
        rms = math.sqrt(mean_S2 - mean_S1**2)
        # print(f"rms: {rms}, mean_S1: {mean_S1}, mean_S2:{mean_S2}, self.S1:{self.S1}, self.S2:{self.S2}, self.N:{self.N}, doc_ids#:{len(self.doc_ids)}")
        return rms

    def get_boundary(self):
        mean, std, z1, z2 = (
            self.calculate_mean(),
            self.calculate_rms(),
            self.z1,
            self.z2,
        )
        # print(f"get_boundary | mean:{mean}, std:{std}, z1:{z1}, z2:{z2}")
        return mean + z1 * std

    def get_distance(self, doc_embs):
        # E_q (batch_size, qlen, 768)
        x_dist = (
            MAX_SCORE
            - calculate_S_qd_regl(doc_embs, self.prototype, doc_embs.device).item()
        )
        return x_dist

    def assign(self, doc_id, doc_embs, doc_hash, ts):
        # E_q (batch_size, qlen, 768)
        # if doc_id not in self.doc_ids:
        self.doc_ids.append(doc_id)
        distance = self.get_distance(doc_embs)
        self.S1 += distance
        self.S2 += distance**2
        self.N = len(self.doc_ids)
        self.prototype += doc_hash
        self.timestamp = ts
        # else:
        #     print(f"Assign duplicate doc id {doc_id}")

    # def evict(
    #     self,
    #     model,
    #     lsh: RandomProjectionLSH,
    #     docs: dict,
    #     required_doc_size,
    #     is_updated,
    # ) -> bool:
    #     temp_docids, temp_qids = [], self.get_only_qids(docs)
    #     # temp_docids, temp_qids = [], []
    #     temp_prototype = torch.zeros_like(self.prototype)
    #     doc_ids = self.get_only_docids(docs)
    #     before_d_n, before_q_n = len(doc_ids), len(temp_qids)
    #     mean, std, z1, z2 = (
    #         self.calculate_mean(),
    #         self.calculate_rms(),
    #         self.z1,
    #         self.z2,
    #     )
    #     BOUNDARY = mean + z2 * std
    #     print(f"BOUNDARY: {BOUNDARY}| mean:{mean}, std:{std}, z1:{z1}, z2:{z2}")

    #     for i in range(0, len(doc_ids), self.batch_size):
    #         batch_doc_ids = doc_ids[i : i + self.batch_size]
    #         batch_key = tuple(batch_doc_ids)
    #         # batch_doc_embs = torch.stack(
    #         #     [docs[doc_id]["TOKEN_EMBS"] for doc_id in batch_doc_ids],
    #         #     dim=0,
    #         # ).squeeze(dim=1)
    #         batch_doc_embs = self.cache.setdefault(
    #             batch_key,
    #             torch.stack(
    #                 [docs[doc_id]["TOKEN_EMBS"] for doc_id in batch_doc_ids], dim=0
    #             ),
    #         )
    #         x_dists = MAX_SCORE - calculate_S_qd_regl_batch(
    #             batch_doc_embs.squeeze(dim=1),
    #             self.prototype.unsqueeze(0),
    #             batch_doc_embs.device,
    #         )
    #         mask = x_dists <= BOUNDARY
    #         selected_indices = torch.nonzero(mask, as_tuple=False).squeeze(dim=1)
    #         if selected_indices.numel() > 0:
    #             selected_doc_ids = [batch_doc_ids[i] for i in selected_indices.tolist()]
    #             selected_texts = [docs[doc_id]["text"] for doc_id in selected_doc_ids]
    #             new_doc_embs = get_passage_embeddings(model, selected_texts).cpu()
    #             temp_docids.extend(selected_doc_ids)
    #             temp_prototype += lsh.encode_batch(
    #                 new_doc_embs
    #             )  # (B, L, D) -> (B, hash_size, D)
    #             # for idx, doc_id in enumerate(selected_doc_ids):
    #             #     doc_emb = new_doc_embs[idx]
    #             #     docs[doc_id]["TOKEN_EMBS"] = doc_emb
    #             #     doc_hash = lsh.encode(doc_emb)
    #             #     temp_docids.append(doc_id)
    #             #     temp_prototype += doc_hash

    #     after_d_n = len(temp_docids)
    #     if after_d_n <= required_doc_size:
    #         return False

    #     qsz = int(len(temp_qids) * after_d_n / before_d_n)
    #     temp_qids = random.sample(temp_qids, qsz)
    #     after_q_n = len(temp_qids)
    #     for i in range(0, len(temp_qids), self.batch_size):
    #         batch_qids = temp_qids[i : i + self.batch_size]
    #         batch_texts = [docs[qid]["text"] for qid in batch_qids]
    #         batch_embs = get_passage_embeddings(model, batch_texts).cpu()
    #         temp_prototype += lsh.encode_batch(
    #             batch_embs
    #         )  # (B, L, D) -> (B, hash_size, D)
    #         # for j, qid in enumerate(batch_qids):
    #         #     doc_emb = batch_embs[j]
    #         #     docs[qid]["TOKEN_EMBS"] = doc_emb
    #         #     doc_hash = lsh.encode(doc_emb)
    #         #     temp_prototype += doc_hash

    #     self.doc_ids = list(set(temp_docids)) + list(set(temp_qids))
    #     # self.doc_ids = list(set(temp_docids))
    #     self.prototype = temp_prototype
    #     print(
    #         f"* Evict result docs {len(temp_docids)}, queries {len(temp_qids)}, total {len(self.doc_ids)}"
    #     )

    #     self.update_statistics(docs)
    #     self.cache = {}
    #     print(
    #         f"doc_ids# {before_d_n} -> {after_d_n},queries# {before_q_n} -> {after_q_n}, new std:{self.calculate_rms()}"
    #     )
    #     return True

    def evict(
        self,
        model,
        lsh: RandomProjectionLSH,
        docs: dict,
        required_doc_size,
        is_updated,
    ) -> bool:
        temp_docids, temp_qids = [], self.get_only_qids(docs)
        temp_prototype = torch.zeros_like(self.prototype)
        doc_ids = self.get_only_docids(docs)
        before_d_n, before_q_n = len(doc_ids), len(temp_qids)
        mean, std, z1, z2 = (
            self.calculate_mean(),
            self.calculate_rms(),
            self.z1,
            self.z2,
        )
        BOUNDARY = mean + z2 * std
        print(f"BOUNDARY: {BOUNDARY}| mean:{mean}, std:{std}, z1:{z1}, z2:{z2}")

        def process_doc_batch(batch_doc_ids):
            batch_key = tuple(batch_doc_ids)
            batch_doc_embs = self.cache.setdefault(
                batch_key,
                torch.stack(
                    [docs[doc_id]["TOKEN_EMBS"] for doc_id in batch_doc_ids], dim=0
                ),
            )
            x_dists = MAX_SCORE - calculate_S_qd_regl_batch(
                batch_doc_embs.squeeze(dim=1),
                self.prototype.unsqueeze(0),
                batch_doc_embs.device,
            )
            mask = x_dists <= BOUNDARY
            selected_indices = torch.nonzero(mask, as_tuple=False).squeeze(dim=1)
            if selected_indices.numel() == 0:
                return [], None
            selected_doc_ids = [batch_doc_ids[i] for i in selected_indices.tolist()]
            selected_texts = [docs[doc_id]["text"] for doc_id in selected_doc_ids]
            new_doc_embs = get_passage_embeddings(model, selected_texts)  # GPU 텐서 유지
            return selected_doc_ids, new_doc_embs

        doc_futures = []
        with ThreadPoolExecutor(max_workers=num_devices) as executor:
            for i in range(0, len(doc_ids), self.batch_size):
                batch_doc_ids = doc_ids[i : i + self.batch_size]
                doc_futures.append(executor.submit(process_doc_batch, batch_doc_ids))

            for f in as_completed(doc_futures):
                selected_doc_ids, new_doc_embs = f.result()
                if selected_doc_ids and new_doc_embs is not None:
                    temp_docids.extend(selected_doc_ids)
                    temp_prototype += lsh.encode_batch(new_doc_embs)

        after_d_n = len(temp_docids)
        if after_d_n <= required_doc_size:
            return False

        qsz = int(len(temp_qids) * after_d_n / before_d_n)
        temp_qids = random.sample(temp_qids, qsz)
        after_q_n = len(temp_qids)

        def process_query_batch(batch_qids):
            batch_texts = [docs[qid]["text"] for qid in batch_qids]
            batch_embs = get_passage_embeddings(model, batch_texts)  # GPU 텐서 유지
            return batch_embs

        q_futures = []
        with ThreadPoolExecutor(max_workers=num_devices) as executor:
            for i in range(0, len(temp_qids), self.batch_size):
                batch_qids = temp_qids[i : i + self.batch_size]
                q_futures.append(executor.submit(process_query_batch, batch_qids))

            for f in as_completed(q_futures):
                batch_embs = f.result()
                if batch_embs is not None:
                    temp_prototype += lsh.encode_batch(batch_embs)

        self.doc_ids = list(set(temp_docids)) + list(set(temp_qids))
        self.prototype = temp_prototype
        self.update_statistics(docs)
        self.cache = {}

        print(
            f"* Evict result docs {len(temp_docids)}, queries {len(temp_qids)}, total {len(self.doc_ids)}"
        )
        print(
            f"doc_ids# {before_d_n} -> {after_d_n}, queries# {before_q_n} -> {after_q_n}, new std:{self.calculate_rms()}"
        )
        return True

    def get_statistics(self):
        return (self.S1, self.S2, self.N)

    def update_statistics(self, docs: dict):
        partition = min(len(self.doc_ids), num_devices)
        doc_ids = self.get_only_docids(docs)
        did_batches = [doc_ids[i::partition] for i in range(partition)]
        q_ids = self.get_only_qids(docs)
        qid_batches = [q_ids[i::partition] for i in range(partition)]
        # print(f"id_batches {id_batches}")

        def process_batch(
            id_batch,
            device,
        ):
            S1, S2 = 0.0, 0.0
            for i in range(0, len(id_batch), self.batch_size):
                batch_token_embs = torch.stack(
                    [
                        docs[doc_id]["TOKEN_EMBS"]
                        for doc_id in id_batch[i : i + self.batch_size]
                    ],
                    dim=0,
                ).squeeze(1)
                # batch_doc_ids = id_batch[i : i + self.batch_size]
                # batch_key = tuple(batch_doc_ids)
                # batch_token_embs = self.cache.setdefault(
                #     batch_key,
                #     torch.stack(
                #         [docs[doc_id]["TOKEN_EMBS"] for doc_id in batch_doc_ids], dim=0
                #     ),
                # )
                # batch_token_embs = batch_token_embs.squeeze(1)
                if self.use_tensor_key:
                    if batch_token_embs.dim() == 2:
                        batch_token_embs = batch_token_embs.unsqueeze(0)
                    print(
                        f"update_statistics batch_token_embs:{batch_token_embs.shape}, self.prototype:{self.prototype.unsqueeze(0).shape}"
                    )
                    score = calculate_S_qd_regl_batch(
                        batch_token_embs, self.prototype.unsqueeze(0), device
                    )
                else:
                    score = calculate_S_qd_regl_dict(
                        batch_token_embs, self.prototype, device
                    )

                x_dist = MAX_SCORE - score
                # print(f"batch_token_embs{batch_token_embs.shape}, x_dist: {x_dist.shape} score:{score.shape}")
                S1 += torch.sum(x_dist).item()
                S2 += torch.sum(x_dist**2).item()
            return (S1, S2)

        results = []
        with ThreadPoolExecutor(max_workers=partition) as executor:
            futures = {
                executor.submit(process_batch, did_batches[i], devices[i]): i
                for i in range(partition)
            }
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
            futures = {
                executor.submit(process_batch, qid_batches[i], devices[i]): i
                for i in range(partition)
            }
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        self.S1 = sum(r[0] for r in results)
        self.S2 = sum(r[1] for r in results)
        self.N = len(self.doc_ids)
        # print(f"update_statistics | S1 {self.S1}, S2 {self.S2}, N {self.N}")

    def get_doc_ids_in_r(self, docs, query, r=None):
        docids_and_scores = self.get_docids_and_scores(
            query, docs
        )  # [(doc_id, score), ...]
        scores = [score for _, score in docids_and_scores]
        if r is None:
            qids = self.get_only_qids(docs)
            score_idx = int(len(scores) * (1 / len(qids)))  # 균등 분할 시 차지하는 갯수
            r = scores[score_idx]
            # r = sum(scores) / len(scores)
        idx = bisect.bisect_left(list(reversed(scores)), r)
        selected = [doc_id for doc_id, _ in docids_and_scores[: len(scores) - idx]]
        return r, selected

    def get_doc_ids_in_r_with_cache(self, cache, docs, query):
        """
        cache: Dict[cluster_id][qid] = List[(doc_id, score)] sorted by score desc
        """
        # 균등 분할 시 차지하는 갯수
        doc_ids = self.get_only_docids(docs)
        qids = self.get_only_qids(docs)
        k = int(len(doc_ids) * (1 / len(qids)))
        docids_and_scores = cache[query["doc_id"]]
        # 왜 캐시사이즈 2*k 보다 훨씬 크지?
        # [cid][qid] q_bsz * 2k 중 해당 클러스터에 속한 문서 수 -> 전체에서 거르고X 클러스터 내에서 거르고
        print(f"cache size({query['doc_id']})/{len(cache[query['doc_id']])}")
        selected = [doc_id for doc_id, _ in docids_and_scores[:k]]
        r = docids_and_scores[k - 1][1]
        return r, selected
