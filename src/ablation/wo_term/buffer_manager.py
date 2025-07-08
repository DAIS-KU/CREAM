from .management import find_k_closest_clusters_for_sampling
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import bisect
import copy
import random

MAX_SCORE = 254


class BufferManager:
    def __init__(self, buffer_size=150):
        self.current_pairs = []
        self.last_pairs = []
        self.buffer_size = buffer_size
        self.sim_sum = 0
        self.pair_n = 0

    def get_all_ids(self):
        pass


class DiversityBufferManager(BufferManager):
    def __init__(self, buffer_size=150):
        super().__init__(buffer_size)

    def get_samples_in_clsuter(self, docs, cluster, qids, sample_size, cache):
        if len(qids) <= sample_size:
            return qids
        else:
            qsets = {}
            for qidx, qid in enumerate(qids):
                query = docs[qid]
                r, doc_ids = cluster.get_doc_ids_in_r_with_cache(cache, docs, query)
                print(f"* {qidx}th query docs in r: #{len(doc_ids)}")
                qsets[qid] = set(doc_ids)

            qsets = dict(
                sorted(qsets.items(), key=lambda item: len(item[1]), reverse=True)
            )
            selected = []
            current_union = set()
            remaining = set(qsets.keys())
            while len(selected) < sample_size and remaining:
                best_qid = None
                best_gain, best_overlap = -1, float("inf")
                for qid in remaining:
                    gain = len(qsets[qid] - current_union)
                    overlap = len(qsets[qid] & current_union)
                    if gain > best_gain or (
                        gain == best_gain and overlap < best_overlap
                    ):
                        best_gain = gain
                        best_overlap = overlap
                        best_qid = qid
                if best_qid is None:
                    break
                print(
                    f"* current_union:{len(current_union)}, best_gain:{best_gain}, best_overlap:{best_overlap}"
                )
                selected.append(best_qid)
                current_union |= qsets[best_qid]
                remaining.remove(best_qid)

            return selected

    def get_samples(self, docs, clusters, caches, sample_size):
        cluster_qids = {}
        total_doc_size, total_query_size = 0, 0
        for cid, cluster in enumerate(clusters):
            qids = cluster.get_only_qids(docs)
            if len(qids):
                cluster_qids[cid] = qids
                total_doc_size += len(cluster.doc_ids)
                total_query_size += len(qids)
        print(f"Total documents {total_doc_size} query {total_query_size}")

        if total_query_size <= sample_size:
            all_qids = []
            for v in cluster_qids.values():
                all_qids.extend(v)
            diverse_qids = copy.deepcopy(all_qids)
            while len(diverse_qids) < sample_size:
                remains = sample_size - len(diverse_qids)
                sample_sz = min(len(all_qids), remains)
                rand_qids = random.sample(all_qids, sample_sz)
                diverse_qids.extend(rand_qids)
        else:
            diverse_qids = set()
            for cid in cluster_qids.keys():
                sample_sz = int(
                    (len(clusters[cid].doc_ids) / total_doc_size) * sample_size
                )
                total_qsz = len(cluster_qids[cid])
                print(f"clusters[{cid}] sampling #{sample_sz} in total #{total_qsz}")
                dqids = self.get_samples_in_clsuter(
                    docs=docs,
                    cluster=clusters[cid],
                    qids=cluster_qids[cid],
                    sample_size=sample_sz,
                    cache=caches[cid],
                )
                diverse_qids.update(dqids)

            while len(diverse_qids) < sample_size:
                remains = sample_size - len(diverse_qids)
                rand_cid = random.sample(list(cluster_qids.keys()), 1)[0]
                sample_sz = min(len(cluster_qids[rand_cid]), remains)
                rand_qids = random.sample(cluster_qids[rand_cid], sample_sz)
                diverse_qids.update(rand_qids)
                print(
                    f"#diverse_qids:{len(diverse_qids)}, remains: {remains}, rand_cid:{rand_cid}, #rand_qids:{len(rand_qids)}"
                )
        trian_queries = [docs[qid] for qid in diverse_qids]
        random.shuffle(trian_queries)
        return trian_queries
