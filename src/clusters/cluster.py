import concurrent
import copy
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

import torch
import math
from functions import (
    calculate_S_qd_regl,
    calculate_S_qd_regl_batch,
    calculate_S_qd_regl_dict,
    get_passage_embeddings,
)

from .prototype import RandomProjectionLSH

MAX_SCORE = 256.0
num_devices = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_devices)]


class Cluster:
    def __init__(
        self,
        model,
        lsh,
        centroid,
        cluster_docs,
        docs: dict,
        use_tensor_key,
        timestamp=0,
        batch_size=384,
        z=2.27,  # 99%
        a=1.0,
        u=0.2,
    ):
        self.lsh = lsh
        self.prototype = centroid
        # print(f"cluster_docs: {type(cluster_docs)}")
        self.doc_ids = [d["doc_id"] for d in cluster_docs]
        self.use_tensor_key = use_tensor_key
        self.timestamp = timestamp
        self.batch_size = batch_size
        self.z = z
        self.u = u
        self.a = a
        # self.cache = {}
        if len(self.doc_ids) == 1:
            self.S1 = 0.0
            self.S2 = 0.0
            self.N = 1
        else:
            self.update_statistics(model, docs)

    # def clear_cache(self):
    #     self.cache = {}

    # def get_cached_emb(self, doc_id, batch_emb):
    #     if doc_id not in self.cache:
    #         doc_emb = get_passage_embeddings(model, docs[doc_id]["text"], device)
    #         self.cache[doc_id]= doc_emb.cpu()
    #     return self.cache[doc_id]

    # def get_cached_embs(self, model, batch_ids, device):
    #     batch_embs = []
    #     for doc_id in batch_ids:
    #         if doc_id not in self.cache:
    #             doc_emb= get_passage_embeddings(model, docs[doc_id]["text"], device)
    #             self.cache[doc_id]= doc_emb.cpu()
    #         batch_embs.append(self.cache[doc_id])
    #     batch_embs = torch.stack(batch_embs)
    #     print(f"get_cached_embs: {batch_embs.shape}")
    #     return batch_embs

    def get_only_docids(self, docs):
        only_doc_ids = [
            doc_id for doc_id in self.doc_ids if not docs[doc_id]["is_query"]
        ]
        print(f"Document only #{len(only_doc_ids)}")
        return only_doc_ids

    # 캐시 들렸다 오는 것 보다 배치 임베딩이 빠를 것 같기도... 개별 순회 vs 배치 연산
    def get_topk_docids_and_scores(self, model, query, docs: dict, k, batch_size=128):
        query_token_embs = get_passage_embeddings(model, query["query"], devices[-1])
        # self.get_cached_emb(query["qid"], query_token_embs)

        def process_batch(device, batch_doc_ids):
            regl_scores = []
            temp_model = copy.deepcopy(model).to(device)
            temp_query_token_embs = query_token_embs.clone().to(device)
            for i in range(0, len(batch_doc_ids), batch_size):
                batch_ids = batch_doc_ids[i : i + batch_size]
                batch_texts = [docs[doc_id]["text"] for doc_id in batch_ids]
                batch_embs = get_passage_embeddings(temp_model, batch_texts, device)
                # batch_embs = self.get_cached_embs(temp_model, batch_ids, device)
                if batch_embs.dim() > 3:
                    batch_embs = batch_embs.squeeze()
                # print(f"get_topk_docids({device}) | batch_embs:{batch_embs.shape}")
                regl_score = calculate_S_qd_regl_batch(
                    temp_query_token_embs, batch_embs, device
                )
                regl_scores.append(regl_score)
                # print(f"regl_scores: {len(regl_scores)}")
            regl_scores = torch.cat(regl_scores, dim=0)
            return [
                (doc_id, regl_scores[idx].item())
                for idx, doc_id in enumerate(batch_doc_ids)
            ]

        regl_scores = []
        # stride, 순서 보장 필요X
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
        top_k_regl_docs = combined_regl_scores[:k]
        return top_k_regl_docs  # [(id, score), ]

    def get_topk_docids(self, model, query, docs: dict, k, batch_size=128) -> List[str]:
        top_k_regl_docs = self.get_topk_docids_and_scores(
            model, query, docs, k, batch_size
        )
        top_k_regl_doc_ids = [x[0] for x in top_k_regl_docs]
        return top_k_regl_doc_ids

    def calculate_mean(self):
        mean = self.S1 / self.N
        return mean

    def calculate_rms(self):
        mean_S1 = self.S1 / self.N
        mean_S2 = self.S2 / self.N
        # print(f"mean_S1: {mean_S1}, mean_S2:{mean_S2}")
        rms = math.sqrt(mean_S2 - mean_S1**2)
        return rms

    def get_boundary(self):
        return self.calculate_mean() + self.z * self.calculate_rms()

    def get_distance(self, doc_embs):
        # E_q (batch_size, qlen, 768)
        if self.use_tensor_key:
            x_dist = MAX_SCORE - calculate_S_qd_regl(
                doc_embs, self.prototype, doc_embs.device
            )
        else:
            x_dist = MAX_SCORE - calculate_S_qd_regl_dict(
                doc_embs, self.prototype, doc_embs.device
            )
        return x_dist.item()

    def assign(self, doc_id, doc_embs, doc_hash, ts):
        # E_q (batch_size, qlen, 768)
        self.doc_ids.append(doc_id)
        distance = self.get_distance(doc_embs)
        self.S1 += distance
        self.S2 += distance**2
        self.N += 1
        if self.use_tensor_key:
            self.prototype += doc_hash
        else:
            for key, value in doc_hash.items():
                self.prototype[key] += value.cpu()
        self.timestamp = ts

    def decay(self):
        print(f"Prototype decayed.")
        self.prototype = self.prototype * math.exp(-self.u)
        # self.clear_cache()

    def evict(
        self, model, lsh: RandomProjectionLSH, docs: dict, required_doc_size
    ) -> bool:  # isNotEmpty
        before_n = len(self.doc_ids)
        temp_docids = []
        temp_prototype = (
            torch.zeros_like(self.prototype)
            if self.use_tensor_key
            else defaultdict(lambda: torch.zeros(768))
        )
        BOUNDARY = self.calculate_mean()  # + self.calculate_rms() * self.a
        print(f"BOUNDARY: {BOUNDARY}")
        for batch_start in range(0, len(self.doc_ids), self.batch_size):
            batch_doc_ids = self.doc_ids[batch_start : batch_start + self.batch_size]
            batch_doc_embs = torch.stack(
                [
                    get_passage_embeddings(model, docs[doc_id]["text"], devices[-1])
                    for doc_id in batch_doc_ids
                ],
                dim=0,
            ).squeeze(dim=1)
            # print(f"evict batch_doc_embs:{batch_doc_embs.shape}")
            x_dists = MAX_SCORE - calculate_S_qd_regl_batch(
                batch_doc_embs, self.prototype.unsqueeze(0), batch_doc_embs.device
            )
            mask = x_dists <= BOUNDARY
            # print(f"evict x_dists:{x_dists.shape}, mask: {mask.shape}")
            for i in range(len(batch_doc_ids)):
                if mask[i].item():
                    temp_docids.append(batch_doc_ids[i])
                    doc_hash = lsh.encode(batch_doc_embs[i])
                    if self.use_tensor_key:
                        temp_prototype += doc_hash
                    else:
                        for key, value in doc_hash.items():
                            temp_prototype[key] += value.cpu()
        self.doc_ids = temp_docids
        self.prototype = temp_prototype
        after_n = len(self.doc_ids)
        print(f"doc_ids# {before_n} -> {after_n}")
        if len(self.doc_ids) < required_doc_size:
            return False
        if before_n != after_n:
            self.update_statistics(model, docs)
        # self.clear_cache()
        return True

    def get_statistics(self):
        return (self.S1, self.S2, self.N)

    def update_statistics(self, model, docs: dict):
        partition = min(len(self.doc_ids), num_devices)
        id_batches = [self.doc_ids[i::partition] for i in range(partition)]
        # print(f"id_batches {id_batches}")

        def process_batch(
            id_batch,
            device,
        ):
            temp_model = copy.deepcopy(model).to(device)
            S1, S2 = 0.0, 0.0
            for i in range(0, len(id_batch), self.batch_size):
                batch_docs = [
                    get_passage_embeddings(temp_model, docs[doc_id]["text"], device)
                    for doc_id in id_batch[i : i + self.batch_size]
                ]
                batch_token_embs = torch.stack(batch_docs, dim=0).squeeze(1)
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
                executor.submit(process_batch, id_batches[i], devices[i]): i
                for i in range(partition)
            }
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        self.S1 = sum(r[0] for r in results)
        self.S2 = sum(r[1] for r in results)
        self.N = len(self.doc_ids)
        # print(f"update_statistics | S1 {self.S1}, S2 {self.S2}, N {self.N}")

    def get_weight(self, T):
        exponent = -(T - self.timestamp) / self.u
        weight = math.exp(exponent)
        return weight

    # def refresh_prototype_with_cache(self):
    #     print(f"Refresh prototype with cached embs")
    #     if self.use_tensor_key:
    #         temp_prototype = torch.zeros_like(self.prototype)
    #         for _id in list(self.cache.keys()):
    #             doc_hash = self.lsh.encode(self.cache[_id])
    #             temp_prototype += doc_hash
    #         self.prototype = temp_prototype
    #     else:
    #         raise NotImplementedError("Unsupported refresh_prototype for sparse hash")
    #     self.clear_cache()
