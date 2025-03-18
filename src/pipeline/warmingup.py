import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from datasets import load_dataset

from functions import (
    renew_data_mean_pooling,
    get_top_k_documents_by_cosine,
    evaluate_dataset,
    InfoNCELoss,
    SimpleContrastiveLoss,
)
from data import read_jsonl, write_file
import time
import random
import json

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, negative_k=6, samples=2500, max_length=256):
        if dataset is None:
            dataset = self.read_jsonl("/mnt/DAIS_NAS/huijeong/pretrained.jsonl")
            print(f"Read {len(dataset)} elements.")
        self.dataset = dataset[:samples]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = samples
        self.negative_k = negative_k

    def read_jsonl(self, filepath):
        res = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                res.append(json.loads(line.strip()))
        return res

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        data = self.dataset[idx]
        query = data["query"]
        positive = data["answer"]
        random_items = random.sample(self.dataset, self.negative_k)
        negatives = [d["answer"] for d in random_items]
        # print(f"query_encoding:{query_encoding['input_ids'].shape}, pos_encoding:{pos_encoding['input_ids'].shape}, neg_encodings:{neg_encodings['input_ids'].shape}")

        return {
            "query_input": query,
            "pos_input": positive,
            "neg_inputs": negatives,
        }


def encode_texts(model, texts, max_length=256):
    device = model.device
    batch_inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
    outputs = model(**batch_inputs).last_hidden_state
    attention_mask = batch_inputs["attention_mask"].unsqueeze(
        -1
    )  # (batch_size, seq_len, 1)

    # Ensure token_embeddings and attention_mask are of the same length
    token_embeddings = outputs[
        :, :max_length, :
    ]  # Limit token embeddings to the max length
    masked_embeddings = token_embeddings * attention_mask  # Element-wise multiplication
    mean_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)

    return mean_embeddings


def train():
    dataset = CustomDataset(None, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.train()

    criterion = InfoNCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    epochs = 1
    i = 1

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_cnt = 1
        # print(f"epoch {epoch}")
        for batch in dataloader:
            # print(f"{i}th batch")
            query_input = batch["query_input"]
            pos_input = batch["pos_input"]
            neg_inputs = batch["neg_inputs"]

            query_emb = encode_texts(model, query_input)
            # query_emb = query_emb.unsqueeze(1)
            pos_emb = encode_texts(model, pos_input)
            pos_emb = pos_emb.unsqueeze(1)
            neg_embs = []
            for neg_input in neg_inputs:
                neg_emb = encode_texts(model, neg_input)
                neg_embs.append(neg_emb)
            neg_embs = torch.stack(neg_embs, dim=1)
            # print(
            #     f"query_emb:{query_emb.shape}, pos_emb:{pos_emb.shape}, neg_embs:{neg_embs.shape}"
            # )
            loss = criterion(query_emb, pos_emb, neg_embs)
            # psg_embs= torch.cat((pos_emb, neg_embs), dim=1)
            # print(
            #     f"query_emb:{query_emb.shape}, psg_embs:{psg_embs.shape}"
            # )
            # loss = criterion(query_emb, psg_embs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Batch {i}, Loss: {loss.item():.4f}, {total_loss/batch_cnt:.4f}")
            i += 1
            batch_cnt += 1
    torch.save(model.state_dict(), "./base_model.pth")


def evaluate(session_cnt=12):
    def model_builder(model_path):
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        model = BertModel.from_pretrained("bert-base-uncased").to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model

    for session_number in range(session_cnt):
        method = "warmingup"
        eval_query_path = (
            f"/mnt/DAIS_NAS/huijeong/sub/test_session{session_number}_queries.jsonl"
        )
        eval_doc_path = (
            f"/mnt/DAIS_NAS/huijeong/sub/test_session{session_number}_docs.jsonl"
        )

        eval_query_data = read_jsonl(eval_query_path, True)
        eval_doc_data = read_jsonl(eval_doc_path, False)

        eval_query_count = len(eval_query_data)
        eval_doc_count = len(eval_doc_data)
        print(f"Query count:{eval_query_count}, Document count:{eval_doc_count}")

        rankings_path = f"../data/rankings/{method}_session_{session_number}.txt"
        model_path = f"../data/model/{method}_session_{session_number}.pth"

        start_time = time.time()
        new_q_data, new_d_data = renew_data_mean_pooling(
            model_builder, "./base_model.pth", eval_query_data, eval_doc_data
        )
        end_time = time.time()
        print(f"Spend {end_time-start_time} seconds for encoding.")

        start_time = time.time()
        result = get_top_k_documents_by_cosine(new_q_data, new_d_data, k=10)
        end_time = time.time()
        print(f"Spend {end_time-start_time} seconds for retrieval.")

        rankings_path = f"../data/rankings/{method}_session_{session_number}.txt"
        write_file(rankings_path, result)
        eval_log_path = f"../data/evals/{method}_{session_number}.txt"
        evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
        del new_q_data, new_d_data
