import json
from collections import defaultdict
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from clusters import RandomProjectionLSH


# =========================
# Bucket Manager (토큰 임베딩용)
# =========================
class BucketManager:
    def __init__(self, store_normalized=True, store_on_cpu=True, dtype=torch.float32):
        """
        store_normalized: 추가 시 미리 정규화 저장(코사인 내적 비용 절감)
        store_on_cpu: 버킷 텐서를 CPU에 저장(대규모 코퍼스 유리)
        dtype: 저장 dtype (float32/float16 권장)
        """
        self.buckets = defaultdict(list)
        self.store_normalized = store_normalized
        self.store_on_cpu = store_on_cpu
        self.dtype = dtype

    def add(self, emb: torch.Tensor, bid):
        """
        emb: [D] 토큰 임베딩 하나
        """
        with torch.no_grad():
            x = emb
            if self.store_normalized:
                x = F.normalize(x, dim=-1)
            x = x.detach().to(
                "cpu" if self.store_on_cpu else emb.device, dtype=self.dtype
            )
            self.buckets[bid].append(x)

    def print_bucket_size(self):
        total = 0
        for bid, embs in self.buckets.items():
            total += len(embs)
            print(f"Bucket {bid} has {len(embs)} token embeddings.")
        print(f"Total token embeddings across all buckets: {total}")

    @torch.no_grad()
    def get_topk_sim(
        self, bid, query_token_emb: torch.Tensor, k=10, chunk_size=4096, device=None
    ):
        """
        동일 버킷(bid)의 문서 토큰들과 쿼리 토큰(query_token_emb) 간 코사인 유사도 top-k 평균
        - 버킷 벡터가 너무 많을 때 chunk_size로 나눠 처리
        - 러닝 top-k 유지(전체 top-k와 동일 결과)
        """
        embs = self.buckets.get(bid, [])
        if not embs:
            return None

        if device is None:
            device = query_token_emb.device

        # 쿼리 정규화(버킷은 add에서 이미 정규화되어 있다 가정)
        q = F.normalize(query_token_emb, dim=-1).to(device)

        running_topk = None
        n = len(embs)
        for i in range(0, n, chunk_size):
            chunk = torch.stack(embs[i : i + chunk_size], dim=0)  # [C, D] (CPU)
            if not self.store_normalized:
                chunk = F.normalize(chunk, dim=-1)
            chunk = chunk.to(device, non_blocking=True)  # GPU 올려 연산
            sims = torch.mv(chunk, q)  # [C]

            if running_topk is None:
                running_topk = sims if sims.numel() <= k else torch.topk(sims, k).values
            else:
                cand = torch.cat([running_topk, sims], dim=0)
                running_topk = cand if cand.numel() <= k else torch.topk(cand, k).values

            # 메모리 정리 힌트
            del chunk, sims

        return running_topk.mean().item() if running_topk is not None else None


# =========================
# 임베딩 유틸 (토큰 임베딩)
# =========================
def encode_token_embs(
    texts,
    model,
    tokenizer,
    device,
    batch_size=256,
    max_length=256,
    exclude_special_tokens=True,
):
    """
    입력 문장 리스트 → 각 문장에 대해 스페셜/패딩 제외 토큰 임베딩 리스트 반환
    return: List[List[Tensor[H]]]  # 문장별 토큰 임베딩 리스트
    """
    all_token_embs_per_text = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            out = model(**enc)  # last_hidden_state: [B, L, H]
            hidden = out.last_hidden_state
            attn = enc["attention_mask"].bool()

        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        input_ids = enc["input_ids"]

        B, L, H = hidden.size()
        for b in range(B):
            # 유효 토큰 (패딩 제거)
            mask = attn[b]  # [L]
            token_ids = input_ids[b][mask]  # [L_valid]
            token_embs = hidden[b][mask]  # [L_valid, H]

            if exclude_special_tokens and (cls_id is not None) and (sep_id is not None):
                keep = (token_ids != cls_id) & (token_ids != sep_id)
                token_embs = token_embs[keep]  # [L_keep, H]

            # 문장별 토큰 임베딩 리스트로 저장
            all_token_embs_per_text.append([tok for tok in token_embs])

    return all_token_embs_per_text


# =========================
# 버킷 빌드 / 유사도 계산
# =========================
def build_document_buckets(
    doc_path,
    model,
    tokenizer,
    lsh,
    device,
    batch_size=256,
    max_length=256,
    bucket_store_normalized=True,
    bucket_store_on_cpu=True,
):
    """
    문서 파일(JSONL, {"text": ...})의 모든 토큰 임베딩을 LSH 버킷에 저장.
    """
    bucket_manager = BucketManager(
        store_normalized=bucket_store_normalized, store_on_cpu=bucket_store_on_cpu
    )

    buffer = []
    d_bcnt = 0

    with open(doc_path, "r", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            buffer.append(text)

            if len(buffer) >= batch_size:
                token_embs_per_text = encode_token_embs(
                    buffer,
                    model,
                    tokenizer,
                    device,
                    batch_size=batch_size,
                    max_length=max_length,
                )
                for token_list in token_embs_per_text:
                    for tok_emb in token_list:
                        bid = lsh._get_key(tok_emb.unsqueeze(0), device, is_list=False)
                        bucket_manager.add(tok_emb, bid)
                buffer = []

            d_bcnt += 1
            if d_bcnt % 100 == 0:
                print(f"Processed {d_bcnt} documents...")

        # 잔여 처리
        if buffer:
            token_embs_per_text = encode_token_embs(
                buffer,
                model,
                tokenizer,
                device,
                batch_size=batch_size,
                max_length=max_length,
            )
            for token_list in token_embs_per_text:
                for tok_emb in token_list:
                    bid = lsh._get_key(tok_emb.unsqueeze(0), device, is_list=False)
                    bucket_manager.add(tok_emb, bid)

    return bucket_manager


def compute_query_similarity(
    query_path,
    model,
    tokenizer,
    lsh,
    bucket_manager,
    device,
    batch_size=256,
    max_length=256,
    k=10,
    chunk_size=4096,
):
    """
    질의 파일(JSONL, {"query": ...})의 각 질의에 대해:
    - 질의 토큰 임베딩을 LSH로 버킷 매핑
    - 같은 버킷의 문서 토큰들과 top-k 코사인 유사도 평균
    - 질의 스코어 = 질의 토큰별 평균을 다시 평균
    """
    similarities = []
    buffer = []
    q_bcnt = 0

    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["query"]
            buffer.append(text)

            if len(buffer) >= batch_size:
                token_embs_per_text = encode_token_embs(
                    buffer,
                    model,
                    tokenizer,
                    device,
                    batch_size=batch_size,
                    max_length=max_length,
                )

                for token_list in token_embs_per_text:
                    token_scores = []
                    for q_tok_emb in token_list:
                        bid = lsh._get_key(
                            q_tok_emb.unsqueeze(0), device, is_list=False
                        )
                        avg_sim = bucket_manager.get_topk_sim(
                            bid, q_tok_emb, k=k, chunk_size=chunk_size, device=device
                        )
                        if avg_sim is not None:
                            token_scores.append(avg_sim)
                    if token_scores:
                        similarities.append(sum(token_scores) / len(token_scores))

                buffer = []

            q_bcnt += 1
            if q_bcnt % 100 == 0:
                print(f"Processed {q_bcnt} queries...")

        # 잔여 처리
        if buffer:
            token_embs_per_text = encode_token_embs(
                buffer,
                model,
                tokenizer,
                device,
                batch_size=batch_size,
                max_length=max_length,
            )
            for token_list in token_embs_per_text:
                token_scores = []
                for q_tok_emb in token_list:
                    bid = lsh._get_key(q_tok_emb.unsqueeze(0), device, is_list=False)
                    avg_sim = bucket_manager.get_topk_sim(
                        bid, q_tok_emb, k=k, chunk_size=chunk_size, device=device
                    )
                    if avg_sim is not None:
                        token_scores.append(avg_sim)
                if token_scores:
                    similarities.append(sum(token_scores) / len(token_scores))

    if similarities:
        overall_avg = sum(similarities) / len(similarities)
        print(
            f"Overall Avg similarity across all queries (token-based): {overall_avg:.4f}"
        )
    else:
        print("No valid similarities computed.")


# =========================
# 엔트리 포인트
# =========================
def get_bucket_sim(
    nbits=10,
    batch_size=256,
    max_length=256,
    k=10,
    chunk_size=4096,
    bucket_store_normalized=True,
    bucket_store_on_cpu=True,
):
    # 디바이스 설정
    num_gpus = torch.cuda.device_count()
    devices = (
        [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        if num_gpus > 0
        else [torch.device("cpu")]
    )
    device = devices[-1]

    # 토크나이저/모델 (로컬 경로)
    tokenizer = BertTokenizer.from_pretrained("/home/work/.default/huijeong/bert_local")
    model = BertModel.from_pretrained("/home/work/.default/huijeong/bert_local")
    model.eval().to(device)

    # LSH 랜덤 투영 벡터 (키 생성용) — CPU 텐서로 두고, 내부에서 필요 시 이동
    random_vectors = torch.randn(nbits, 768)  # [nbits, dim]
    lsh = RandomProjectionLSH(
        random_vectors=random_vectors, embedding_dim=768, use_tensor_key=True
    )

    # 문서 버킷 생성 (토큰 임베딩 저장)
    doc_buckets = build_document_buckets(
        "/home/work/.default/huijeong/data/msmarco_session/train_session0_docs_filtered.jsonl",
        model,
        tokenizer,
        lsh,
        device,
        batch_size=batch_size,
        max_length=max_length,
        bucket_store_normalized=bucket_store_normalized,
        bucket_store_on_cpu=bucket_store_on_cpu,
    )

    doc_buckets.print_bucket_size()

    # 질의-문서 유사도 계산 (토큰 기준)
    compute_query_similarity(
        "/home/work/.default/huijeong/data/msmarco_session/train_session0_queries.jsonl",
        model,
        tokenizer,
        lsh,
        doc_buckets,
        device,
        batch_size=batch_size,
        max_length=max_length,
        k=k,
        chunk_size=chunk_size,
    )
