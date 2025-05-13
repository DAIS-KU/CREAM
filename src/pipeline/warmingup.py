import json
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from transformers import BertModel, BertTokenizer

from data import read_jsonl, read_jsonl_as_dict, write_file
from functions import (
    InfoNCELoss,
    SimpleContrastiveLoss,
    evaluate_dataset,
    get_top_k_documents_by_cosine,
    renew_data_mean_pooling,
    get_top_k_documents,
)
from clusters import renew_data


class LOTTEDataset(Dataset):
    def __init__(self, tokenizer, max_samples=200, split="train", max_length=512):
        self.queries = {}
        self.passages = {}
        for domain in ["technology", "writing"]:  # "science", "lifestyle", "recreation"
            _q = read_jsonl_as_dict(
                f"../data/raw/lotte/{domain}/{domain}_queries.jsonl", "qid"
            )
            self.queries.update(_q)
            _d = read_jsonl_as_dict(
                f"../data/raw/lotte/{domain}/{domain}_docs.jsonl", "doc_id"
            )
            self.passages.update(_d)

        for session_number in range(9):
            _t = read_jsonl_as_dict(
                f"../data/datasetD/train_session{session_number}_queries.jsonl", "qid"
            )
            keys_to_remove = list(_t.keys())
            [self.queries.pop(key, None) for key in keys_to_remove]
            _t = read_jsonl_as_dict(
                f"../data/datasetD/test_session{session_number}_queries.jsonl", "qid"
            )
            keys_to_remove = list(_t.keys())
            [self.queries.pop(key, None) for key in keys_to_remove]
        self.queries = list(self.queries.values())
        print(f"Read LOTTE queries {len(self.queries)}, passages {len(self.passages)}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.indexes = range(len(self.queries))
        self.max_samples = min(max_samples, len(self.queries))

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        item = self.queries[idx]
        query = item["query"]
        pos_passage_id = item["answer_pids"][0]
        pos_passage = self.passages[pos_passage_id]["text"]

        random_indices = random.sample(self.indexes, 6)
        negative_ids = [self.queries[_id]["answer_pids"][0] for _id in random_indices]
        neg_passages = [self.passages[_id]["text"] for _id in negative_ids]

        return {
            "query_enc": query,
            "pos_enc": pos_passage,
            "neg_enc": neg_passages,
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]  # None 제거
    return default_collate(batch)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class MSMARCODataset(Dataset):
    def __init__(self, tokenizer, split="train", max_length=512):
        self.data = load_dataset("microsoft/ms_marco", "v1.1")[split]
        print(f"Read MSMARCO {len(self.data)}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.indexes = range(len(self.data))
        self.keys_to_remove = set()
        for session_number in range(12):
            _t = read_jsonl_as_dict(
                f"../data/datasetD/ms_marco/train_session{session_number}_queries.jsonl",
                "qid",
            )
            keys_to_remove = update(_t.keys())
            _t = read_jsonl_as_dict(
                f"../data/datasetD/ms_marco/test_session{session_number}_queries.jsonl",
                "qid",
            )
            keys_to_remove = update(_t.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["query"]
        if len(item["answers"]) == 0 or item["query_id"] in self.keys_to_remove:
            return None

        pos_passages = item["answers"][0]
        random_indices = random.sample(self.indexes, 12)
        negatives = []
        for _id in random_indices:
            if len(self.data[_id]["answers"]) > 0:
                negatives.extend(self.data[_id]["answers"])
            if len(negatives) > 6:
                break

        return {
            "query_enc": query,
            "pos_enc": pos_passages,
            "neg_enc": negatives[:6],
        }


class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, negative_k=6, samples=2000, max_length=256):
        if dataset is None:
            dataset = self.read_jsonl("../data/pretrained.jsonl")
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
    dataset = LOTTEDataset(tokenizer)
    # # dataset = MSMARCODataset(tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True, drop_last=True, collate_fn=collate_fn
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
            if batch_cnt == 1800:
                break

            # print(f"{i}th batch")
            # query_input = batch["query_input"]
            # pos_input = batch["pos_input"]
            # neg_inputs = batch["neg_inputs"]

            query_input, pos_input, neg_inputs = (
                batch["query_enc"],
                batch["pos_enc"],
                batch["neg_enc"],
            )

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
    torch.save(model.state_dict(), "../data/base_model_lotte.pth")


def evaluate_cosine(session_cnt=9):
    def model_builder(model_path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = BertModel.from_pretrained("/home/work/retrieval/bert-base-uncased").to(
            device
        )
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model

    method = "pretrained_cosine"
    model_path = f"../data/base_model_lotte.pth"
    for session_number in range(session_cnt):
        eval_query_path = f"../data/datasetD/test_session{session_number}_queries.jsonl"
        eval_doc_path = f"../data/datasetD/test_session{session_number}_docs.jsonl"

        eval_query_data = read_jsonl(eval_query_path, True)
        eval_doc_data = read_jsonl(eval_doc_path, False)

        eval_query_count = len(eval_query_data)
        eval_doc_count = len(eval_doc_data)
        print(f"Query count:{eval_query_count}, Document count:{eval_doc_count}")

        rankings_path = f"../data/rankings/{method}_session_{session_number}.txt"

        start_time = time.time()
        new_q_data, new_d_data = renew_data_mean_pooling(
            model_builder,
            model_path,
            eval_query_data,
            eval_doc_data,  # "./base_model.pth"
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


def evaluate_term(sesison_count=10):
    method = "pretrained_term"
    model_path = None  # f"../data/base_model_lotte.pth"
    for session_number in range(sesison_count):
        eval_query_path = f"../data/datasetD/test_session{session_number}_queries.jsonl"
        eval_doc_path = f"../data/datasetD/test_session{session_number}_docs.jsonl"

        eval_query_data = read_jsonl(eval_query_path, True)
        eval_doc_data = read_jsonl(eval_doc_path, False)

        eval_query_count = len(eval_query_data)
        eval_doc_count = len(eval_doc_data)
        print(f"Query count:{eval_query_count}, Document count:{eval_doc_count}")

        rankings_path = f"../data/rankings/{method}_session_{session_number}.txt"

        start_time = time.time()
        new_q_data, new_d_data = renew_data(
            queries=eval_query_data,
            documents=eval_doc_data,
            model_path=model_path,
            nbits=12,
            renew_q=True,
            renew_d=True,
            use_tensor_key=True,
        )
        end_time = time.time()
        print(f"Spend {end_time-start_time} seconds for encoding.")

        start_time = time.time()
        result = get_top_k_documents(new_q_data, new_d_data)
        end_time = time.time()
        print(f"Spend {end_time-start_time} seconds for retrieval.")

        rankings_path = f"../data/rankings/{method}_session_{session_number}.txt"
        write_file(rankings_path, result)
        eval_log_path = f"../data/evals/{method}_{session_number}.txt"
        evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
        del new_q_data, new_d_data
