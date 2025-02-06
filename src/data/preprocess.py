import torch
from concurrent.futures import ThreadPoolExecutor
from cluster import RandomProjectionLSH
from transformers import BertModel, BertTokenizer

from concurrent.futures import ProcessPoolExecutor

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
    model, lsh, query_batch, device, batch_size=2048, max_length=256
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
            model.load_state_dict(torch.load(model_path))
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
    print("Query-Document encoding ended.")

    new_q_data = {}
    new_d_data = {}
    for result in results:
        new_q_data.update(result[0])
        new_d_data.update(result[1])

    return new_q_data, new_d_data
