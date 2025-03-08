import torch
from concurrent.futures import ThreadPoolExecutor
from cluster import RandomProjectionLSH
from transformers import BertModel, BertTokenizer

import time
from buffer import (
    Buffer,
    DataArguments,
    TevatronTrainingArguments,
    DenseModel,
    ModelArguments,
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


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


def _renew_queries_with_text(
    model, lsh, query_batch, device, batch_size=2048 + 1024, max_length=256
):
    torch.cuda.set_device(device)
    query_texts = [q["query"] for q in query_batch]
    query_ids = [q["qid"] for q in query_batch]
    query_embeddings, query_hashes, query_decoded_texts = [], [], []
    for i in range(0, len(query_texts), batch_size):
        print(f"{device} | Query encoding batch {i}")
        query_batch_text = query_texts[i : i + batch_size]
        (
            query_batch_embeddings,
            query_batch_decoded_texts,
        ) = get_passage_embeddings_with_text(
            model, query_batch_text, device, max_length
        )
        for query_batch_embedding in query_batch_embeddings:
            query_embeddings.append(query_batch_embedding.cpu())
            query_hashes.append(lsh.encode(query_batch_embedding))
        # print(f'query_hashes: {query_hashes[-1][list(query_hashes[-1].keys())[-1]].device}, query_embeddings: {query_embeddings[-1].device}')
        query_decoded_texts.extend(query_batch_decoded_texts)

    new_q_data = {
        qid: {"ID": qid, "TEXT": text, "LSH_MAPS": maps, "TOKEN_EMBS": emb}
        for qid, text, maps, emb in zip(
            query_ids, query_decoded_texts, query_hashes, query_embeddings
        )
    }
    del query_ids, query_decoded_texts, query_embeddings, query_hashes
    return new_q_data


def _renew_queries(model, lsh, query_batch, device, batch_size=2048, max_length=256):
    torch.cuda.set_device(device)
    query_texts = [q["query"] for q in query_batch]
    query_ids = [q["qid"] for q in query_batch]
    query_embeddings, query_hashes = [], []
    for i in range(0, len(query_texts), batch_size):
        print(f"{device} | Query encoding batch {i}")
        query_batch_text = query_texts[i : i + batch_size]
        query_batch_embeddings = get_passage_embeddings(
            model, query_batch_text, device, max_length
        )
        for query_batch_embedding in query_batch_embeddings:
            query_embeddings.append(query_batch_embedding.cpu())
            query_hashes.append(lsh.encode(query_batch_embedding))

    new_q_data = {
        qid: {"ID": qid, "LSH_MAPS": maps, "TOKEN_EMBS": emb}
        for qid, maps, emb in zip(query_ids, query_hashes, query_embeddings)
    }
    del query_ids, query_embeddings, query_hashes
    return new_q_data


def _renew_docs_with_text(
    model, lsh, document_batch, device, batch_size=2048, max_length=256
):
    torch.cuda.set_device(device)
    document_texts = [d["text"] for d in document_batch]
    document_ids = [d["doc_id"] for d in document_batch]
    document_embeddings, document_hashes, document_decoded_texts = [], [], []
    for i in range(0, len(document_texts), batch_size):
        print(f"{device} | Document encoding batch {i}")
        doc_batch_text = document_texts[i : i + batch_size]
        (
            doc_batch_embeddings,
            doc_batch_decoded_texts,
        ) = get_passage_embeddings_with_text(model, doc_batch_text, device, max_length)
        for doc_batch_embedding in doc_batch_embeddings:
            document_embeddings.append(doc_batch_embedding.cpu())
            document_hashes.append(lsh.encode(doc_batch_embedding))
        document_decoded_texts.extend(doc_batch_decoded_texts)

    new_d_data = {
        doc_id: {"ID": doc_id, "TEXT": text, "LSH_MAPS": maps, "TOKEN_EMBS": emb}
        for doc_id, text, maps, emb in zip(
            document_ids, document_decoded_texts, document_hashes, document_embeddings
        )
    }
    del document_ids, document_decoded_texts, document_embeddings, document_hashes
    return new_d_data


def _renew_docs(model, lsh, document_batch, device, batch_size=2048, max_length=256):
    torch.cuda.set_device(device)
    document_texts = [d["text"] for d in document_batch]
    document_ids = [d["doc_id"] for d in document_batch]
    document_embeddings, document_hashes = [], []
    for i in range(0, len(document_texts), batch_size):
        print(f"{device} | Document encoding batch {i}")
        doc_batch_text = document_texts[i : i + batch_size]
        doc_batch_embeddings = get_passage_embeddings(
            model, doc_batch_text, device, max_length
        )
        for doc_batch_embedding in doc_batch_embeddings:
            document_embeddings.append(doc_batch_embedding.cpu())
            document_hashes.append(lsh.encode(doc_batch_embedding))

    new_d_data = {
        doc_id: {"ID": doc_id, "LSH_MAPS": maps, "TOKEN_EMBS": emb}
        for doc_id, maps, emb in zip(document_ids, document_hashes, document_embeddings)
    }
    del document_ids, document_embeddings, document_hashes
    return new_d_data


def _renew_data(
    model,
    lsh,
    query_batch,
    document_batch,
    device,
    renew_q=True,
    renew_d=True,
    batch_size=2048,
    max_length=256,
):
    torch.cuda.set_device(device)
    print(
        f"Starting on {device} with {len(query_batch)} queries and {len(document_batch)} documents (batch size {batch_size})"
    )
    new_q_data = (
        _renew_queries(model, lsh, query_batch, device, batch_size, max_length)
        if renew_q
        else {}
    )
    new_d_data = (
        _renew_docs(model, lsh, document_batch, device, batch_size, max_length)
        if renew_d
        else {}
    )
    return new_q_data, new_d_data


def renew_data(
    queries,
    documents,
    nbits,
    embedding_dim,
    model_path=None,
    renew_q=True,
    renew_d=True,
):
    num_gpus = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
    print(f"Using {num_gpus} GPUs: {devices}")

    models, hashes = [], []
    random_vectors = torch.randn(nbits, embedding_dim)
    for device in devices:
        model = BertModel.from_pretrained("bert-base-uncased").to(device)
        if model_path is not None:
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.eval()
        models.append(model)
        hashes.append(
            RandomProjectionLSH(
                random_vectors=random_vectors, embedding_dim=embedding_dim
            )
        )

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
                hashes,
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


def renew_data_mean_pooling(queries_data, documents_data, model_path):
    def build_model(model_path=None):
        model_args = ModelArguments(model_name_or_path="bert-base-uncased")
        training_args = TevatronTrainingArguments(output_dir="../data/model")
        model = DenseModel.build(
            model_args,
            training_args,
            cache_dir=model_args.cache_dir,
        )
        if model_path:
            model.load_state_dict(torch.load(model_path, weights_only=True))
        return model

    num_gpus = torch.cuda.device_count()
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
