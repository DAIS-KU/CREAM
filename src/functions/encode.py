from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import BertTokenizer
from buffer import (
    DataArguments,
    ModelArguments,
    TevatronTrainingArguments,
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def get_passage_embeddings(model, passages, device, max_length=256):
    model.to(device)
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


def process_batch(
    batch_data,
    model,
    tokenizer,
    device_id,
    id_field="qid",
    text_field="query",
    batch_size=512,
):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    results = {}

    for i in range(0, len(batch_data), batch_size):
        print(f" process_batch({device}) {i}/{len(batch_data)}")
        batch_chunk = batch_data[i : i + batch_size]  # batch_size만큼 자르기

        item_ids = [item[id_field] for item in batch_chunk]
        texts = [item[text_field] for item in batch_chunk]

        encoded_input = tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = model.encode_mean_pooling(
                encoded_input
            )  # [batch_size, emb_dim]

        for item_id, emb in zip(item_ids, model_output.cpu()):
            results[item_id] = {
                "ID": item_id,
                "EMB": emb.cpu(),
            }
    return results


def renew_data_mean_pooling(model_builder, queries_data, documents_data):
    num_gpus = 3  # torch.cuda.device_count()
    models = [build_model(model_path) for _ in range(num_gpus)]
    query_batches = []
    doc_batches = []

    for i in range(num_gpus):
        query_start = i * len(queries_data) // num_gpus
        query_end = (i + 1) * len(queries_data) // num_gpus
        query_batches.append(queries_data[query_start:query_end])

        doc_start = i * len(documents_data) // num_gpus
        doc_end = (i + 1) * len(documents_data) // num_gpus
        doc_batches.append(documents_data[doc_start:doc_end])

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        print(f"Query mean-pooling embedding starts.")
        futures = []
        for i in range(num_gpus):
            if query_batches[i]:
                futures.append(
                    executor.submit(
                        process_batch,
                        query_batches[i],
                        models[i],
                        tokenizer,
                        i,
                        "qid",
                        "query",
                    )
                )
        query_results = {}
        for future in futures:
            query_results.update(future.result())

        print(f"Document mean-pooling embedding starts..")
        futures = []
        for i in range(num_gpus):
            if doc_batches[i]:
                futures.append(
                    executor.submit(
                        process_batch,
                        doc_batches[i],
                        models[i],
                        tokenizer,
                        i,
                        "doc_id",
                        "text",
                    )
                )
        doc_results = {}
        for future in futures:
            doc_results.update(future.result())
    return query_results, doc_results
