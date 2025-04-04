import concurrent
import copy
import math
from concurrent.futures import ThreadPoolExecutor
from typing import List

import torch
import torch.nn.functional as F

from functions import calculate_S_qd_regl, encode_texts_mean_pooling

MAX_SCORE = 1.0
num_devices = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_devices)]


class Cluster:
    def __init__(
        self,
        model,
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
        self.prototype = centroid
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

    def get_only_docids(self, docs):
        only_doc_ids = [
            doc_id for doc_id in self.doc_ids if not docs[doc_id]["is_query"]
        ]
        print(f"Document only #{len(only_doc_ids)}")
        return only_doc_ids

    def get_topk_docids_and_scores(self, model, query, docs: dict, k, batch_size=128):
        query_token_embs = encode_texts_mean_pooling(model, query["query"])

        def process_batch(device, batch_doc_ids):
            scores = []
            temp_model = copy.deepcopy(model).to(device)
            temp_query_token_embs = query_token_embs.clone().to(device)
            for i in range(0, len(batch_doc_ids), batch_size):
                batch_ids = batch_doc_ids[i : i + batch_size]
                batch_texts = [docs[doc_id]["text"] for doc_id in batch_ids]
                batch_embs = encode_texts_mean_pooling(temp_model, batch_texts)

                if batch_embs.dim() > 3:
                    batch_embs = batch_embs.squeeze()

                batch_scores = F.cosine_similarity(
                    temp_query_token_embs, batch_embs.to(temp_query_token_embs), dim=-1
                )
                scores.append(batch_scores)

            scores = torch.cat(scores, dim=0)
            return [
                (doc_id, scores[idx].item()) for idx, doc_id in enumerate(batch_doc_ids)
            ]

        scores = []
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
                scores.extend(future.result())

        combined_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_k_docs = combined_scores[:k]
        return top_k_docs  # [(doc_id, score), ...]

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
        # print(f"doc_embs:{doc_embs.shape}, self.prototype:{self.prototype.shape}")
        x_dist = MAX_SCORE - F.cosine_similarity(
            doc_embs, self.prototype.to(doc_embs.device), dim=-1
        )
        return x_dist.item()

    def assign(self, doc_id, doc_embs, ts):
        # E_q (batch_size, qlen, 768)
        self.doc_ids.append(doc_id)
        distance = self.get_distance(doc_embs)
        self.S1 += distance
        self.S2 += distance**2
        self.N += 1
        self.prototype += doc_embs.to(self.prototype.device)
        self.timestamp = ts

    def evict(self, model, docs: dict, required_doc_size) -> bool:
        before_n = len(self.doc_ids)
        temp_docids = []
        temp_prototype = torch.zeros_like(self.prototype)

        BOUNDARY = self.calculate_mean()
        print(f"BOUNDARY: {BOUNDARY}")

        for batch_start in range(0, len(self.doc_ids), self.batch_size):
            batch_doc_ids = self.doc_ids[batch_start : batch_start + self.batch_size]
            batch_doc_texts = [docs[doc_id]["text"] for doc_id in batch_doc_ids]
            batch_doc_embs = encode_texts_mean_pooling(model, batch_doc_texts)
            print(f"batch_doc_embs: {batch_doc_embs.shape}")

            if batch_doc_embs.dim() == 3:  # (batch, 1, embedding_dim)
                batch_doc_embs = batch_doc_embs.squeeze(dim=1)

            cos_sim = F.cosine_similarity(
                batch_doc_embs,
                self.prototype.unsqueeze(0).to(batch_doc_embs.device),
                dim=-1,
            )
            mask = cos_sim >= BOUNDARY

            for i in range(len(batch_doc_ids)):
                if mask[i].item():
                    temp_docids.append(batch_doc_ids[i])
                    doc_hash = batch_doc_embs[i]
                    temp_prototype += doc_hash

        self.doc_ids = temp_docids
        self.prototype = temp_prototype / len(self.doc_ids)
        after_n = len(self.doc_ids)

        print(f"doc_ids# {before_n} -> {after_n}")

        if len(self.doc_ids) < required_doc_size:
            return False

        if before_n != after_n:
            self.update_statistics(model, docs)

        return True

    def get_statistics(self):
        return (self.S1, self.S2, self.N)

    def update_statistics(self, model, docs: dict):
        partition = min(len(self.doc_ids), num_devices)
        id_batches = [self.doc_ids[i::partition] for i in range(partition)]

        def process_batch(id_batch, device):
            temp_model = copy.deepcopy(model).to(device)
            S1, S2 = 0.0, 0.0
            for i in range(0, len(id_batch), self.batch_size):
                batch_doc_texts = [
                    docs[doc_id]["text"] for doc_id in id_batch[i : i + self.batch_size]
                ]
                batch_token_embs = encode_texts_mean_pooling(
                    temp_model, batch_doc_texts
                )
                # print(f"update_statistics-process_batch {batch_token_embs.shape}")

                if batch_token_embs.dim() == 2:
                    batch_token_embs = batch_token_embs.unsqueeze(0)
                score = F.cosine_similarity(
                    batch_token_embs,
                    self.prototype.unsqueeze(0).to(batch_token_embs.device),
                    dim=-1,
                )

                x_dist = MAX_SCORE - score
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

    def get_weight(self, T):
        exponent = -(T - self.timestamp) / self.u
        weight = math.exp(exponent)
        return weight
