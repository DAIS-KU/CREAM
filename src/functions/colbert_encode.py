import time
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F


from typing import Dict
import torch
from transformers import BertModel, BertTokenizerFast

from buffer import (
    Buffer,
    DataArguments,
    DenseModel,
    ModelArguments,
    TevatronTrainingArguments,
)

# RuntimeError: Already borrowed 각 스레드별로 생성해서 사용


def get_tokenizer():
    tokenizer = BertTokenizerFast.from_pretrained(
        "/home/work/retrieval/bert-base-uncased/bert-base-uncased"
    )
    # ColBERT 논문에 등장하는 [Q], [D] 두 토큰을 신규 등록
    special_tokens_dict = {"additional_special_tokens": ["[Q]", "[D]"]}
    num_new = tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


def encode_query(
    query: str,
    tokenizer,
    Nq: int = 32,
) -> Dict[str, torch.Tensor]:
    """
    ColBERT용 쿼리 인코딩 함수
    - [CLS] + [Q] + query tokens + [MASK]*padding
    - 전체 길이 Nq + 2 (CLS, Q 포함)
    """
    num_special_tokens = 2  # [CLS], [Q]

    tokens = tokenizer.tokenize(query)
    tokens = tokens[: Nq - 2]  # 특수 토큰 포함 최대 Nq개로 제한
    pad_len = Nq - 2 - len(tokens)

    input_tokens = ["[CLS]", "[Q]"] + tokens + ["[MASK]"] * pad_len
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),  # (Nq,)
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0),
    }


def encode_document(
    doc: str,
    tokenizer,
    Nd: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    ColBERT용 문서 인코딩 함수
    - [CLS] [D] <tokens> 구조
    - 전체 길이 Nd로 강제 고정
    - tokenizer.encode_plus로 자동 truncation + padding
    """
    encoding = tokenizer.encode_plus(
        "[D] " + doc,
        add_special_tokens=True,  # adds [CLS], [SEP]
        max_length=Nd,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return encoding  # {input_ids, attention_mask}


def get_query_embeddings(model, queries, tokenizer, device=None, max_length=32):
    device = model.device if device is None else device
    model.to(device)
    batch_inputs = encode_query(queries, tokenizer)
    batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
    with torch.no_grad():
        outputs = model(query=batch_inputs)
        q_token_embs = outputs.q_reps
        pad_count = max_length - q_token_embs[0].shape[0]
        if pad_count:
            dim = q_token_embs[0].shape[1]
            padding = torch.zeros(pad_count, dim, device=q_token_embs[0].device)
            # print(f"q_token_embs[0]: {q_token_embs[0].shape}, padding: {padding.shape}")
            q_emb = torch.cat(q_token_embs + [padding], dim=0)
        else:
            q_emb = q_token_embs[0]
    return q_emb


def get_passage_embeddings(model, passages, tokenizer, device=None, max_length=128):
    device = model.device if device is None else device
    model.to(device)
    batch_inputs = encode_document(passages, tokenizer)
    batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
    with torch.no_grad():
        outputs = model(passage=batch_inputs)
        p_token_embs = outputs.p_reps
        pad_count = max_length - p_token_embs[0].shape[0]
        if pad_count:
            dim = p_token_embs[0].shape[1]
            padding = torch.zeros(pad_count, dim, device=p_token_embs[0].device)
            # print(f"p_token_embs[0]: {p_token_embs[0].shape}, padding: {padding.shape}")
            p_emb = torch.cat(p_token_embs + [padding], dim=0)
        else:
            p_emb = p_token_embs[0]
    return p_emb


def _renew_queries(model, query_batch, device, batch_size=3072, max_length=32):
    torch.cuda.set_device(device)
    tokenizer = get_tokenizer()
    query_texts = [q["query"] for q in query_batch]
    query_ids = [q["qid"] for q in query_batch]
    query_answers = [q["answer_pids"] for q in query_batch]
    query_embeddings = []
    for i, query_text in enumerate(query_texts):
        query_embedding = get_query_embeddings(
            model, query_text, tokenizer, device, max_length
        )
        if i % 10 == 0:
            print(f"{device} | Query encoding {i} / {query_embedding.shape}")
        query_embeddings.append(query_embedding)
        # print(f"_renew_queries | {query_embeddings[0].shape}")
    new_q_data = {
        qid: {
            "doc_id": qid,
            "text": text,
            "TOKEN_EMBS": emb,
            "is_query": True,
            "answer_pids": pids,
        }
        for qid, text, emb, pids in zip(
            query_ids, query_texts, query_embeddings, query_answers
        )
    }
    print(
        f"new_q_data | new_q_data:{len(list(new_q_data.keys()))}, query_embeddings:{len(query_embeddings)}"
    )
    del query_ids, query_embeddings
    torch.cuda.empty_cache()
    return new_q_data


def _renew_docs(model, document_batch, device, batch_size=3072, max_length=128):
    torch.cuda.set_device(device)
    tokenizer = get_tokenizer()
    document_texts = [d["text"] for d in document_batch]
    document_ids = [d["doc_id"] for d in document_batch]
    document_embeddings = []
    for i, document_text in enumerate(document_texts):
        doc_embedding = get_passage_embeddings(
            model, document_text, tokenizer, device, max_length
        )
        if i % 100 == 0:
            print(f"{device} | Document encoding {i}/ {doc_embedding.shape}")
        document_embeddings.append(doc_embedding)
        # print(f"_renew_docs | {document_embeddings[0].shape}")

    new_d_data = {
        doc_id: {
            "doc_id": doc_id,
            "text": text,
            "TOKEN_EMBS": emb,
            "is_query": False,
            "answer_pids": [],
        }
        for doc_id, text, emb in zip(document_ids, document_texts, document_embeddings)
    }
    print(
        f"new_d_data | new_d_data:{len(list(new_d_data.keys()))}, document_embeddings:{len(document_embeddings)}"
    )
    del document_ids, document_embeddings
    torch.cuda.empty_cache()
    return new_d_data


def _renew_data(
    model,
    query_batch,
    document_batch,
    device,
    renew_q=True,
    renew_d=True,
    batch_size=3072,
):
    torch.cuda.set_device(device)
    print(
        f"Starting on {device} with {len(query_batch)} queries and {len(document_batch)} documents (batch size {batch_size})"
    )
    new_q_data = (
        _renew_queries(model, query_batch, device, batch_size) if renew_q else {}
    )
    new_d_data = (
        _renew_docs(model, document_batch, device, batch_size) if renew_d else {}
    )
    return new_q_data, new_d_data


def renew_data(
    queries,
    documents,
    model_builder,
    model_path,
    renew_q=True,
    renew_d=True,
):
    num_gpus = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
    print(f"Using {num_gpus} GPUs: {devices}")

    models = []
    for device in devices:
        model = model_builder(model_path=model_path)
        model.eval()
        models.append(model)

    query_batches = (
        [queries[i::num_gpus] for i in range(num_gpus)]
        if renew_q
        else [[None] for _ in range(num_gpus)]
    )
    document_batches = (
        [documents[i::num_gpus] for i in range(num_gpus)]
        if renew_d
        else [[None] for _ in range(num_gpus)]
    )

    print("Query-Document encoding started.")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        results = list(
            executor.map(
                _renew_data,
                models,
                query_batches,
                document_batches,
                devices,
                [renew_q for _ in range(num_gpus)],
                [renew_d for _ in range(num_gpus)],
            )
        )
    end_time = time.time()
    print(f"Query-Document encoding ended.({end_time-start_time} sec.)")

    new_q_data = {}
    new_d_data = {}
    for result in results:
        new_q_data.update(result[0])
        new_d_data.update(result[1])

    return new_q_data, new_d_data
