import random
import torch
from transformers import BertModel, BertTokenizer

torch.autograd.set_detect_anomaly(True)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def encode_texts(model, texts, max_length=256):
    device = model.device
    no_padding_inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    no_padding_inputs = {
        key: value.to(device) for key, value in no_padding_inputs.items()
    }
    outputs = model(**no_padding_inputs).last_hidden_state
    embedding = outputs[:, 0, :]  # [CLS]만 사용
    return embedding


def session_train(queries, docs, method, num_epochs):
    random.shuffle(queries)
    query_cnt = len(queries)
    loss_values = []
    validation_values = []
    accuracy_values = []

    current_device_index = torch.cuda.current_device()
    device = torch.device(f"cuda:{current_device_index}")

    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.train()
    model_path = f"../data/model/ground_truth_in_batch.pth"

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
            batch_pos_docs = []  # 현재 batch의 모든 positive 문서 저장

            for qid in range(start_idx, end_idx):
                query = queries[qid]
                pos_doc = docs[random.sample(query["answer_pids"], 1)[0]]["text"]
                batch_pos_docs.append(pos_doc)
                pos_docs_batch.append(pos_doc)

            for qid in range(start_idx, end_idx):
                query = queries[qid]
                pos_docs = [batch_pos_docs[qid - start_idx]]
                # batch 내 다른 positive 문서 중 3개 샘플링 (현재 query의 positive 문서는 제외)
                available_neg_docs = [doc for doc in batch_pos_docs if doc != pos_doc]
                neg_docs = random.sample(
                    available_neg_docs, min(3, len(available_neg_docs))
                )

                query_batch.append(query["query"])
                pos_embeddings = encode_texts(
                    model=model,
                    texts=pos_docs,
                )  # (positive_k, embedding_dim)
                pos_docs_batch.append(pos_embeddings)
                neg_embeddings = encode_texts(
                    model=model,
                    texts=neg_docs,
                )  # (negative_k, embedding_dim)
                neg_docs_batch.append(neg_embeddings)

            query_embeddings = encode_texts(
                model=model,
                texts=query_batch,
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

    return loss_values


def train(
    sesison_count=1,
    num_epochs=1,
    include_evaluate=True,
):
    for session_number in range(sesison_count):
        print(f"Training Session {session_number}")
        model_path = f"../data/model/gt_session_{session_number}.pth"

        # 새로운 세션 문서
        doc_path = f"../data/sessions/train_session{session_number}_docs.jsonl"
        doc_data = read_jsonl(doc_path)[:500]
        _, doc_data = renew_data(
            queries=None,
            documents=doc_data,
            nbits=16,
            embedding_dim=768,
            renew_q=False,
            renew_d=True,
        )
        print(f"Session {session_number} | Document count:{len(doc_data)}")

        # 샘플링 및 대조학습 수행
        model = BertModel.from_pretrained("bert-base-uncased").to(devices[0])
        if session_number != 0:
            model.load_state_dict(torch.load(model_path))
        model.train()

        loss_values = session_train(
            query_path=f"../data/sessions/train_session{session_number}_queries.jsonl",
            model=model,
            num_epochs=num_epochs,
        )
        total_loss_values.append(loss_values)
        show_loss(total_loss_values)
        torch.save(model.state_dict(), model_path)


def evaluate(sesison_count=1):
    for session_number in range(sesison_count):
        print(f"Evaluate Session {session_number}")
        eval_query_path = f"../data/sessions/test_session{session_number}_queries.jsonl"
        eval_doc_path = f"../data/sessions/test_session{session_number}_docs.jsonl"

        eval_query_data = read_jsonl(query_path)
        eval_doc_data = read_jsonl(doc_path)

        eval_query_count = len(eval_query_data)
        eval_doc_count = len(eval_doc_data)
        print(f"Query count:{query_count}, Document count:{doc_count}")

        rankings_path = f"../data/rankings/gt_{session_number}.txt"
        model_path = f"../data/model/gt_session_{session_number}.pth"
        new_q_data, new_d_data = renew_data(
            queries=eval_query_data,
            documents=eval_doc_data,
            nbits=0,
            embedding_dim=768,
            model_path=model_path,
            renew_q=True,
            renew_d=True,
        )
        result = get_top_k_documents(new_q_data, new_d_data, k=10)

        rankings_path = f"../data/rankings/gt_{session_number}.txt"
        write_file(rankings_path, result)
        evaluate_dataset(method, domain, query_path, rankings_path, doc_count)
        del new_q_data, new_d_data
