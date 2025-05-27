from functions import calculate_S_qd_regl_batch_batch
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
                print(f"* {qidx}th query docs in r {r}: #{len(doc_ids)}")
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
                rand_cid = random.sample(cluster_qids.keys(), 1)[0]
                sample_sz = min(len(cluster_qids[rand_cid]), remains)
                rand_qids = random.sample(cluster_qids[rand_cid], sample_sz)
                diverse_qids.update(rand_qids)
                print(
                    f"#diverse_qids:{len(diverse_qids)}, remains: {remains}, rand_cid:{rand_cid}, #rand_qids:{len(rand_qids)}"
                )
        trian_queries = [docs[qid] for qid in diverse_qids]
        random.shuffle(trian_queries)
        return trian_queries


class GreedyDiversityBufferManager(BufferManager):
    def __init__(self, buffer_size=150):
        super().__init__(buffer_size)

    def update_current_info(self, qids):
        self.current_pairs = qids

    def get_samples(self, docs, sample_size):
        return self.get_diverse_queries_by_distance(docs, sample_size)

    def get_dist_sum(self, docs, res_pairs, cand):
        candidate_pairs = [cand] + list(res_pairs)
        candidate_embs = [docs[qid]["TOKEN_EMBS"] for qid in candidate_pairs]
        N = len(candidate_embs)

        all_embs = torch.stack(candidate_embs)
        num_devices = torch.cuda.device_count()
        num_chunks = min(N, num_devices)

        chunk_sizes = [(N + i) // num_chunks for i in range(num_chunks)]
        chunk_starts = [sum(chunk_sizes[:i]) for i in range(num_chunks)]
        chunk_indices = [
            list(range(start, start + size))
            for start, size in zip(chunk_starts, chunk_sizes)
        ]
        sim_matrix = torch.zeros((N, N), device="cuda:0")

        def compute_chunk(chunk_idx_list, device_id):
            device = f"cuda:{device_id}"
            all_embs_device = all_embs.to(device)
            partial_result = torch.zeros((len(chunk_idx_list), N), device=device)
            for j, i in enumerate(chunk_idx_list):
                i_emb = all_embs_device[i].unsqueeze(0)  # [1, seq_len, 768]
                scores = MAX_SCORE - calculate_S_qd_regl_batch_batch(
                    i_emb, all_embs_device, device
                )  # [1, N]
                partial_result[j] = scores.squeeze(0)
            return chunk_idx_list, partial_result.to("cuda:0")

        futures = []
        with ThreadPoolExecutor(max_workers=num_chunks) as executor:
            for device_id, chunk in enumerate(chunk_indices):
                futures.append(executor.submit(compute_chunk, chunk, device_id))
        # 결과를 원래 순서에 맞게 sim_matrix에 반영
        for future in futures:
            chunk_idx_list, partial_result = future.result()
            for j, i in enumerate(chunk_idx_list):
                sim_matrix[i] = partial_result[j]
        # 자기 자신을 제외한 최소 거리/유사도 계산 TODO 멀티5도메인일 때 최소 유사도로 츄라이해보기
        dist_matrix = MAX_SCORE - sim_matrix
        # dist_matrix = sim_matrix
        for i in range(N):
            dist_matrix[i, i] = float("inf")
        min_vals = torch.min(dist_matrix, dim=1).values
        dist_sum = min_vals.sum().item()
        return dist_sum

    # def get_dist_sum(self, docs, res_pairs, cand):
    #     candidate_pairs = [cand] + list(res_pairs)
    #     candidate_embs = [docs[qid]['TOKEN_EMBS'] for qid in candidate_pairs]
    #     N = len(candidate_pairs)
    #     # Step 1: 전체 유사도 행렬 계산
    #     all_embs = torch.stack(candidate_embs).to('cuda:0')  # [N, seq_len, 768]
    #     sim_matrix = torch.zeros((N, N), device='cuda:0')
    #     for i in range(N):
    #         i_emb = all_embs[i].unsqueeze(0)
    #         scores = calculate_S_qd_regl_batch_batch(i_emb, all_embs, 'cuda:0')  # shape: (1, N)
    #         sim_matrix[i] = scores.squeeze(0)
    #     # Step 2: 각 row마다 자기 자신 제외하고 min 값 구함
    #     for i in range(N):
    #         sim_matrix[i, i] = float('inf')  # 자기 자신 제외
    #     min_vals = torch.min(sim_matrix, dim=1).values
    #     dist_sum = min_vals.sum().item()
    #     return dist_sum

    def get_diverse_queries_by_distance(self, docs, sample_size):
        candidate_pairs = self.current_pairs + self.last_pairs
        if len(candidate_pairs) <= sample_size:
            self.last_pairs = self.current_pairs + self.last_pairs
            self.current_pairs = []
            return candidate_pairs
        res_pairs = set([candidate_pairs.pop()])
        # 최근접 쿼리간 거리합이 최대화되는 쿼리를 추가
        prev_dist_sum = 0
        tmp_candidate_pairs = copy.deepcopy(candidate_pairs)
        trial = 0
        # 하는 거 보고 제한 결정
        while len(res_pairs) < sample_size:
            for i, cand in enumerate(tmp_candidate_pairs):
                dist_sum = self.get_dist_sum(docs, res_pairs, cand)
                if prev_dist_sum < dist_sum:
                    res_pairs.add(cand)
                    candidate_pairs.remove(cand)
                    prev_dist_sum = dist_sum
                print(f"** {i}th candidate done.({len(res_pairs)})")
                if len(res_pairs) == sample_size:
                    break
            print(
                f"* {trial}th trial is done. dist_sum: {prev_dist_sum}/({len(res_pairs)})"
            )
            trial += 1
        # # 부족하면 남은 것 중 랜덤 추가
        # if len(res_pairs) < sample_size:
        #     remains_size = sample_size - len(res_pairs)
        #     remains = random.sample(candidate_pairs, remains_size)
        #     res_pairs.update(remains)
        res_pairs = list(res_pairs)

        self.last_pairs = copy.deepcopy(res_pairs)
        self.current_pairs = []
        print(f"get_diverse_queries_by_distance last_pairs:{len(self.last_pairs)}")
        return res_pairs

    def get_all_ids(self):
        # print(f"DiversityBufferManager.all_qid_pids: {self.last_pairs }")
        return self.last_pairs
