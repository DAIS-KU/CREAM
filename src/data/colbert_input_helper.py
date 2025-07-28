import torch
from typing import Dict
import string
import random
import numpy as np
from .bm25 import BM25Okapi, preprocess
from .loader import load_train_docs, read_jsonl, read_jsonl_as_dict
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained(
    "/home/work/retrieval/bert-base-uncased/bert-base-uncased"
)
# ColBERT 논문에 등장하는 [Q], [D] 두 토큰을 신규 등록
special_tokens_dict = {"additional_special_tokens": ["[Q]", "[D]"]}
num_new = tokenizer.add_special_tokens(special_tokens_dict)


def encode_query(
    query: str,
    tokenizer: BertTokenizerFast,
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
    pad_len = Nq - len(tokens)

    input_tokens = ["[CLS]", "[Q]"] + tokens + ["[MASK]"] * pad_len
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),  # (Nq,)
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def encode_document(
    doc: str,
    tokenizer: BertTokenizerFast,
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
        return_tensors=None,
    )

    return {
        "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
    }


def get_documents_for_query(docs, query, bm25, doc_ids, positive_k, negative_k):
    pos_doc_id = random.choice(query["cos_ans_pids"])
    # pos_doc_id = random.choice(query["answer_pids"])
    pos_doc = docs[pos_doc_id]["text"]
    tokenized_query = preprocess(query["query"].lower())
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = np.argpartition(-scores, negative_k * 4)[: negative_k * 4]
    ranked_indices = ranked_indices[np.argsort(-scores[ranked_indices])]
    answer_pids = set(query["cos_ans_pids"])
    # answer_pids = set(query["answer_pids"])
    neg_doc_ids = [doc_ids[i] for i in ranked_indices if doc_ids[i] not in answer_pids][
        :negative_k
    ]
    neg_docs = [docs[pid]["text"] for pid in neg_doc_ids]
    return [pos_doc] + neg_docs


def build_bm25(docs):
    tokenized_corpus = [preprocess(doc["text"].lower()) for doc in docs]
    bm25 = BM25Okapi(corpus=tokenized_corpus, k1=0.8, b=0.75)
    return bm25


def prepare_inputs(session_number, positive_k=1, negative_k=6):
    inputs = {}
    query_path = (
        f"../data/datasetL_large/train_session{session_number}_queries_cos.jsonl"
    )
    doc_path = f"../data/datasetL_large/train_session{session_number}_docs.jsonl"
    queries = read_jsonl(query_path, is_query=True)
    docs = read_jsonl_as_dict(doc_path, id_field="doc_id")
    doc_ids = list(docs.keys())
    bm25 = build_bm25(list(docs.values()))
    for qid, query in enumerate(queries):
        print(f"Prepare {qid}th query.")
        query_tensor = encode_query(query["query"], tokenizer)
        query_docs = get_documents_for_query(
            docs, query, bm25, doc_ids, positive_k, negative_k
        )
        print()
        doc_tensors = [encode_document(doc, tokenizer) for doc in query_docs]

        # batchify document tensors
        doc_input_ids = torch.stack(
            [d["input_ids"] for d in doc_tensors], dim=0
        )  # (D, Nd)
        doc_attention = torch.stack([d["attention_mask"] for d in doc_tensors], dim=0)

        inputs[qid] = (
            {
                "input_ids": query_tensor["input_ids"].unsqueeze(0),  # (1, Nq)
                "attention_mask": query_tensor["attention_mask"].unsqueeze(0),
            },
            {"input_ids": doc_input_ids, "attention_mask": doc_attention},  # (D, Nd)
        )
    return inputs
