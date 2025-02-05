import random
import torch
from transformers import BertModel, BertTokenizer

torch.autograd.set_detect_anomaly(True)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def encode_texts(model, texts, device, max_length=256, is_cls=True):
    no_padding_inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    no_padding_inputs = {
        key: value.to(device) for key, value in no_padding_inputs.items()
    }
    outputs = model(**no_padding_inputs).last_hidden_state
    if is_cls:
        embedding = outputs[:, 0, :]  # [CLS]만 사용
    else:
        embedding = outputs[0, 1:-1, :]  # [CLS]와 [SEP] 제외
    return embedding


def train(queries, docs, num_epochs):
    random.shuffle(queries)
    query_cnt = len(queries)
    loss_values = []
    validation_values = []
    accuracy_values = []

    current_device_index = torch.cuda.current_device()
    device = torch.device(f"cuda:{current_device_index}")

    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.train()
    model_path = f"../data/model/random_sampling.pth"

    loss_fn = InfoNCELoss()
    learning_rate = 2e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    batch_size = 32

    for epoch in range(num_epochs):
        total_loss = 0

        for start_idx in range(0, query_cnt, batch_size):
            end_idx = min(start_idx + batch_size, query_cnt)
            print(f"batch {start_idx}-{end_idx}")

            query_batch, pos_docs_batch, neg_docs_batch = [], [], []

            for qid in range(start_idx, end_idx):
                query = queries[qid]
                pos_docs = [
                    doc["text"] for doc in random.sample(list(docs.values()), 3)
                ]
                neg_docs = [
                    doc["text"] for doc in random.sample(list(docs.values()), 3)
                ]
                query_batch.append(query["query"])
                pos_embeddings = encode_texts(
                    model=model,
                    texts=pos_docs,
                    device=device,
                    max_length=256,
                    is_cls=True,
                )  # (positive_k, embedding_dim)
                pos_docs_batch.append(pos_embeddings)
                neg_embeddings = encode_texts(
                    model=model,
                    texts=neg_docs,
                    device=device,
                    max_length=256,
                    is_cls=True,
                )  # (negative_k, embedding_dim)
                neg_docs_batch.append(neg_embeddings)

            query_embeddings = encode_texts(
                model=model,
                texts=query_batch,
                device=device,
                max_length=256,
                is_cls=True,
            )  # (batch_size, embedding_dim)
            positive_embeddings = torch.stack(
                pos_docs_batch
            )  # (batch_size, positive_k, embedding_dim)
            negative_embeddings = torch.stack(
                neg_docs_batch
            )  # (batch_size, negative_k, embedding_dim)
            print(
                f"query: {query_embeddings.shape}, pos: {positive_embeddings.shape} | negs: {negative_embeddings.shape}"
            )

            loss = loss_fn(query_embeddings, positive_embeddings, negative_embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss_values.append(loss.item())
            print(
                f"Processed {end_idx}/{query_cnt} queries | Batch Loss: {loss.item():.4f} | Total Loss: {total_loss / ((end_idx + 1) // batch_size):.4f}"
            )

            # 클러스터링 시에는 클러스터 계산 후 .cpu() / .to(device)로 필요한 클러스터의 인스턴스만 gpu 활용
            del query_embeddings, positive_embeddings, negative_embeddings
            torch.cuda.empty_cache()

            torch.save(model.state_dict(), model_path)
            success_k, recall_k = do_expermient(method=method, mode="val")
            validation_values.append((success_k, recall_k))
            success_k, recall_k = do_expermient(method=method, mode="training_accuracy")
            accuracy_values.append((success_k, recall_k))

    show_loss(loss_values)
    show_success_recall(accuracy_values)
    show_success_recall(validation_values)
