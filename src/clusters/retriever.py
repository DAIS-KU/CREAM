import torch
from concurrent.futures import ThreadPoolExecutor

from transformers import BertTokenizer
from .management import find_closest_cluster_id
from functions import calculate_S_qd_regl_batch
from collections import defaultdict


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

num_gpus = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]


def get_passage_embeddings(model, passages, device, max_length=256):
    batch_inputs = tokenizer(
        passages,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
    with torch.no_grad():
        outputs = model(**batch_inputs).last_hidden_state
    token_embeddings = outputs[:, 1:-1, :]
    attention_mask = batch_inputs["attention_mask"][:, 1:-1]
    token_embeddings = token_embeddings * (attention_mask[:, :, None].to(device))
    return token_embeddings


def retrieve_top_k_docs_from_cluster(model, queries, clusters, docs, k=10):
    result = {}
    for query in queries:
        closest_cluster_id = find_k_closest_cluster(model, query["query"], clusters, 1)
        closest_cluster = clusters[closest_cluster_id]
        top_k_doc_ids = closest_cluster.get_topk_docids(model, query, docs, k)
        {query["qid"]: top_k_doc_ids}
    return result
