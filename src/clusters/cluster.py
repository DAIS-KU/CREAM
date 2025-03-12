import concurrent
import copy
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

import torch

from functions import (
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
        centroid,
        cluster_docs,
        docs: dict,
        timestamp=0,
        batch_size=256,
        z=1.96,
        u=5,
    ):
        self.prototype = centroid
        self.doc_ids = [d["doc_id"] for d in cluster_docs]
        self.timestamp = timestamp
        self.batch_size = batch_size
        self.z = z
        self.u = u
        if len(self.doc_ids) == 1:
            self.S1 = 0.0
            self.S2 = 0.0
            self.N = 1
        else:
            self.update_statistics(model, docs)

    def get_topk_docids(self, model, query, docs: dict, k) -> List[str]:
        query_token_embs = get_passage_embeddings(model, query["query"], devices[-1])
        regl_scores = []
        for i in range(0, len(self.doc_ids), self.batch_size):
            batch_doc_ids = self.doc_ids[i : i + self.batch_size]
            batch_docs = [
                get_passage_embeddings(
                    model, docs[doc_id]["text"], devices[-1]
                ).squeeze()
                for doc_id in batch_doc_ids
            ]
            combined_embs = torch.stack(batch_docs, dim=0)
            # print(f"query_token_embs: {query_token_embs.shape}, combined_embs:{combined_embs.shape}")
            regl_score = calculate_S_qd_regl_batch(
                query_token_embs, combined_embs, devices[-1]
            )
            # print(f"regl_score: {regl_score.shape}")
            regl_scores.extend(
                [
                    (doc_id, regl_score[idx].item())
                    for idx, doc_id in enumerate(batch_doc_ids)
                ]
            )
        combined_regl_scores = sorted(regl_scores, key=lambda x: x[1], reverse=True)
        top_k_regl_docs = combined_regl_scores[:k]
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

    def get_boundary(self, z=1.96):
        return self.calculate_mean() + self.z * self.calculate_rms()

    def get_distance(self, doc_embs):
        # E_q (batch_size, qlen, 768)
        doc_embs = doc_embs.unsqueeze(0)
        x_dist = MAX_SCORE - calculate_S_qd_regl_dict(
            doc_embs, self.prototype, doc_embs.device
        )
        return x_dist.item()

    def assign(self, doc_id, doc_embs, doc_hash, ts):
        self.doc_ids.append(doc_id)
        # E_q (batch_size, qlen, 768)
        doc_embs = doc_embs.unsqueeze(0)
        distance = MAX_SCORE - calculate_S_qd_regl_dict(
            doc_embs, self.prototype, doc_embs.device
        )
        self.S1 += distance.item()
        self.S2 += distance.item() ** 2
        self.N += 1
        for key, value in doc_hash.items():
            self.prototype[key] += value.cpu()
        self.timestamp = ts

    def evict(self, model, lsh: RandomProjectionLSH, docs: dict) -> bool:  # isNotEmpty
        temp_docids, temp_doc_hashes = [], []
        BOUNDARY = self.get_boundary()
        before_n = len(self.doc_ids)
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
            x_dists = MAX_SCORE - calculate_S_qd_regl_dict(
                batch_doc_embs, self.prototype, batch_doc_embs.device
            )
            mask = x_dists <= BOUNDARY
            # print(f"evict x_dists:{x_dists.shape}, mask: {mask.shape}")
            for i in range(len(batch_doc_ids)):
                if mask[i].item():
                    temp_docids.append(batch_doc_ids[i])
                    temp_doc_hashes.append(lsh.encode(batch_doc_embs[i]))
        if len(temp_docids) == 0:
            return False
        self.doc_ids = temp_docids
        after_n = len(self.doc_ids)
        self.prototype = defaultdict(lambda: torch.zeros(768))
        for doc_hash in temp_doc_hashes:
            for key, value in doc_hash.items():
                self.prototype[key] += value.cpu()
        self.update_statistics(model, docs)
        # print(f"doc_ids# {before_n} -> {after_n}")
        return True

    def get_statistics(self):
        return (self.S1, self.S2, self.N)

    def update_statistics(self, model, docs: dict):
        partition = min(len(self.doc_ids), num_devices)
        id_batches = [self.doc_ids[i::num_devices] for i in range(partition)]
        # print(f"id_batches {id_batches}")

        def process_batch(
            id_batch,
            device,
        ):
            temp_model = copy.deepcopy(model)
            S1, S2 = 0.0, 0.0
            for i in range(0, len(id_batch), self.batch_size):
                batch_docs = [
                    get_passage_embeddings(temp_model, docs[doc_id]["text"], device)
                    for doc_id in id_batch[i : i + self.batch_size]
                ]
                batch_token_embs = torch.stack(batch_docs, dim=0).squeeze(1)
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
