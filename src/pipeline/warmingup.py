import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from datasets import load_dataset

from functions import renew_data_mean_pooling, get_top_k_documents_by_cosine


class MSMARCODataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset[:3000]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        query = sample["query"]
        query_id = sample["query_id"]
        query_type = sample["query_type"]
        positive = (
            sample["passages"]["passage_text"][
                sample["passages"]["is_selected"].index(1)
            ]
            if 1 in sample["passages"]["is_selected"]
            else ""
        )
        negative = (
            sample["passages"]["passage_text"][
                sample["passages"]["is_selected"].index(0)
            ]
            if 0 in sample["passages"]["is_selected"]
            else ""
        )

        pos_encoding = self.tokenizer(
            query,
            positive,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        neg_encoding = self.tokenizer(
            query,
            negative,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "query_id": query_id,
            "query_type": query_type,
            "pos_input": pos_encoding,
            "neg_input": neg_encoding,
        }


class ContrastiveBERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.bert.config.hidden_size, 128)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(cls_embedding)
        return projected


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, query_emb, pos_emb, neg_emb):
        pos_sim = self.cosine_similarity(query_emb, pos_emb)
        neg_sim = self.cosine_similarity(query_emb, neg_emb)
        loss = -torch.log(
            torch.exp(pos_sim / self.temperature)
            / (
                torch.exp(pos_sim / self.temperature)
                + torch.exp(neg_sim / self.temperature)
            )
        )
        return loss.mean()


def train():
    dataset = load_dataset("microsoft/ms_marco", "v2.1", split="train")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = MSMARCODataset(dataset, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveBERT().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    epochs = 1
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            pos_input = batch["pos_input"]
            neg_input = batch["neg_input"]

            input_ids_pos = pos_input["input_ids"].squeeze(1).to(device)
            attention_mask_pos = pos_input["attention_mask"].squeeze(1).to(device)
            input_ids_neg = neg_input["input_ids"].squeeze(1).to(device)
            attention_mask_neg = neg_input["attention_mask"].squeeze(1).to(device)

            query_emb = model(input_ids_pos, attention_mask_pos)
            pos_emb = model(input_ids_pos, attention_mask_pos)
            neg_emb = model(input_ids_neg, attention_mask_neg)

            loss = criterion(query_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Batch {i+1}, Loss: {total_loss/len(dataloader):.4f}")
    torch.save(model.state_dict(), "./base_model.pth")


def evaluate(session_cnt=12):
    def model_builder(model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased").to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model
    for i in range(session_cnt):
        eval_query_path = f"../data/sub/test_session{session_number}_queries.jsonl"
        eval_doc_path = f"../data/sub/huijeong/test_session{session_number}_docs.jsonl"

        eval_query_data = read_jsonl(eval_query_path)
        eval_doc_data = read_jsonl(eval_doc_path)

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


if __name__ == "__main__":
    train()
