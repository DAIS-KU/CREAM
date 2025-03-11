from prototype import RandomProjectionLSH
from dataclasses import dataclass
from functions import calculate_S_qd_regl_batch
from collections import defaultdict
from retrieval import get_passage_embeddings

num_devices = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_devices)]


class Cluster:
    def __init__(self, centroid, cluster_docs, batch_size=256, timestamp=0):
        self.prototype = centroid
        self.doc_ids = [d["doc_id"] for d in cluster_docs]
        self.ts = timestamp
        self.batch_size = batch_size
        self.update_statistics()

    def get_topk_docids(self, model, query, docs, k) -> List[str]:
        query_token_embs = get_passage_embeddings(query["query"])
        for i in range(0, len(self.doc_ids), self.batch_size):
            batch_docs = [
                get_passage_embeddings(model, docs[doc_id]["text"])
                for doc_id in doc_ids[i : i + self.batch_size]
            ]
            combined_embs = torch.stack(batch_docs, dim=0)
            regl_score = calculate_S_qd_regl_batch(
                query_token_embs.unsqueeze(dim=0), combined_embs, device
            )
            regl_scores.extend(
                [
                    (doc["doc_id"], regl_score[idx].item())
                    for idx, doc in enumerate(batch_docs)
                ]
            )
        combined_regl_scores = sorted(regl_scores, key=lambda x: x[1], reverse=True)
        top_k_regl_docs = combined_regl_scores[:k]
        top_k_regl_doc_ids = [x[0] for x in top_k_regl_docs]
        return top_k_regl_doc_ids

    def calculate_mean(self, S1, N):
        mean = self.S1 / self.N
        return mean

    def calculate_rms(self, S1, S2, N):
        mean_S1 = self.S1 / N
        mean_S2 = self.S2 / N
        rms = math.sqrt(mean_S2 - mean_S1**2)
        return rms

    def get_boundary(self, z=1.96):
        return self.calculate_mean() + z * calculate_rms()

    def get_distance(self, doc_embs):
        x_dist = MAX_SCORE - calculate_S_qd_regl_dict(
            doc_embs, self.prototype, doc_embs.device
        )
        return x_dist.item()

    def assign(self, doc_id, doc_embs, doc_hash, ts):
        self.doc_ids.append(doc_id)
        distance = MAX_SCORE - calculate_S_qd_regl_dict(
            doc_embs, self.prototype, doc_embs.device
        )
        self.S1 += distance.item()
        self.S2 += distance.item() ** 2
        self.N += 1
        for key, value in doc_hash.items():
            self.prototype[key] += value.cpu()
        self.timestamp = ts

    def evict(self, model, lsh, docs) -> bool:
        temp_docids, temp_doc_hashes = [], []
        BOUNDARY = self.get_boundary()
        for doc_batch_start in range(0, len(self.doc_ids), self.batch_size):
            batch_doc_ids = doc_ids[batch_start : batch_start + self.batch_size]
            batch_doc_embs = torch.stack(
                [
                    get_passage_embeddings(model, docs[doc_id]["text"])
                    for doc_id in batch_doc_ids
                ],
                dim=0,
            )
            x_dists = MAX_SCORE - calculate_S_qd_regl_dict(
                batch_doc_embs, self.prototype, batch_doc_embs.device
            )
            mask = x_dists <= BOUNDARY
            for i in range(len(batch_doc_ids)):
                if mask[i]:
                    temp_docids.append(batch_docids[i])
                    temp_doc_hashes.append(lsh.encode(batch_doc_embs[i]))
        if len(temp_docids) == 0:
            return False
        self.doc_ids = temp_docids()
        self.prototype = defaultdict(lambda: torch.zeros(768))
        for doc_hash in temp_doc_hashes:
            for key, value in doc_hash.items():
                self.prototype[key] += value.cpu()
        self.update_statistics(docs)
        return True

    def get_statistics(self):
        return (self.S1, self.S2, self.N)

    def update_statistics(self, docs):
        id_batches = [doc_ids[i::num_devices] for i in range(num_devices)]

        def process_batch(
            id_batch,
            docs,
            device,
        ):
            for i in range(0, len(batch), self.batch_size):
                batch_docs = [
                    get_passage_embeddings(docs[doc_id]["text"])
                    for doc_id in id_batch[i : i + self.batch_size]
                ]
                batch_token_embs = torch.stack(batch_docs, dim=0)
                x_dist = MAX_SCORE - calculate_S_qd_regl_dict(
                    batch_token_embs, self.prototype, device
                )
                S1 += torch.sum(x_dist)
                S2 += torch.sum(x_dist**2)
            return (S1.item(), S2.item())

        results = []
        with ThreadPoolExecutor(max_workers=num_devices) as executor:
            futures = {
                executor.submit(process_batch, batches[i], devices[i]): i
                for i in range(num_devices)
            }
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        self.S1 = sum(r[0] for r in results)
        self.S2 = sum(r[1] for r in results)
        self.N = len(self.doc_ids)

        print(f"{self.id} update_statistics | S1 {self.S1}, S2 {self.S2}, N {self.N}")
