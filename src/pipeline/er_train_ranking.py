import random
import torch
from transformers import BertTokenizer
from functions import (
    SimpleContrastiveLoss,
    evaluate_dataset,
    get_top_k_documents,
    evaluate_by_cosine,
)

from data import read_jsonl, write_file, renew_data, prepare_inputs
import time
from buffer import (
    Buffer,
    DataArguments,
    TevatronTrainingArguments,
    DenseModel,
    ModelArguments,
)

torch.autograd.set_detect_anomaly(True)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

num_gpus = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]


def build_buffer():
    query_data = f"/mnt/DAIS_NAS/huijeong/train_session0_queries.jsonl"
    doc_data = f"/mnt/DAIS_NAS/huijeong/train_session0_docs.jsonl"
    # buffer_data = "../data"
    output_dir = "../data"

    method = "er"
    model_args = ModelArguments(model_name_or_path="bert-base-uncased")
    training_args = TevatronTrainingArguments(output_dir="../data/model")
    model = DenseModel.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
    )
    # model_path = f"../data/model/{method}_session_0.pth"
    # model.load_state_dict(torch.load(model_path))
    buffer = Buffer(
        model,
        tokenizer,
        DataArguments(
            query_data=query_data,
            doc_data=doc_data,
            # buffer_data=buffer_data,
            mem_batch_size=3,
        ),
        TevatronTrainingArguments(output_dir=output_dir),
    )
    return buffer


def get_top_k_documents_by_cosine(query_emb, current_data_embs, k=10):
    scores = torch.nn.functional.cosine_similarity(
        query_emb.unsqueeze(0), current_data_embs, dim=1
    )
    top_k_scores, top_k_indices = torch.topk(scores, k)
    return top_k_indices.tolist()


# https://github.com/caiyinqiong/L-2R/blob/main/src/tevatron/trainer.py
def session_train(inputs, model, num_epochs):
    input_cnt = len(inputs)
    print(f"input_cnt: {input_cnt}")
    # 이미 배치처리가 되어있어서 쌓아서 할 필요없나(네)
    # (q_lst, d_lst, identity, old_emb) List[Dict[str, Union[torch.Tensor, Any]]]
    # q_lst = ([qid], [p1n7 doc_id], {'input_ids': ...}, {'attention_mask': ...})
    # q_lst[2] torch.Size([1, 32]), q_lst[3] torch.Size([8, 128])
    loss_values = []
    loss_fn = SimpleContrastiveLoss()
    learning_rate = 2e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    batch_size = 8  # 32

    for epoch in range(num_epochs):
        total_loss = 0

        for start_idx in range(0, input_cnt, batch_size):
            end_idx = min(start_idx + batch_size, input_cnt)
            print(f"batch {start_idx}-{end_idx}")

            x_batch, y_batch = [], []
            for _id in range(start_idx, end_idx):
                x_embeddings = model(inputs[_id][2])
                y_embeddings = model(inputs[_id][3])
                x_batch.append(x_embeddings)
                y_batch.append(y_embeddings)
            # EncoderOutput(loss=loss, scores=scores, q_reps=q_reps, p_reps=p_reps)의 loss를 사용해도 되는건가???
            # SimpleContrastiveLoss(x: Tensor, y: Tensor, ...)이면 loss_fn(x=q_reps, y=p_reps)로 사용해야 하는건가???
            x_batch_tensor = torch.stack(x_embeddings)
            y_batch_tensor = torch.stack(y_embeddings)
            print(
                f"x_batch_tensor: {x_batch_tensor.shape}, y_batch_tensor: {y_batch_tensor.shape}"
            )
            loss = loss_fn(x_batch_tensor, y_batch_tensor)

            optimizer.zero_grad()
            loss.backward()
            # del input_batch
            optimizer.step()

            total_loss += loss.item()
            loss_values.append(loss.item())

            print(
                f"Processed {end_idx}/{input_cnt} queries | Batch Loss: {loss.item():.4f} | Total Loss: {total_loss / ((end_idx + 1) // batch_size):.4f}"
            )
    return loss_values


def train(session_count=1, num_epochs=1):
    buffer = build_buffer()
    method = "er"
    for session_number in range(session_count):
        print(f"Train Session {session_number}")
        query_path = (
            f"/mnt/DAIS_NAS/huijeong/train_session{session_number}_queries.jsonl"
        )
        doc_path = f"/mnt/DAIS_NAS/huijeong/train_session{session_number}_docs.jsonl"
        inputs = prepare_inputs(query_path, doc_path, buffer, method)

        model_args = ModelArguments(model_name_or_path="bert-base-uncased")
        output_dir = "../data/model"
        training_args = TevatronTrainingArguments(output_dir=output_dir)
        model = DenseModel.build(
            model_args,
            training_args,
            cache_dir=model_args.cache_dir,
        )
        if session_number != 0:
            model_path = f"../data/model/{method}_session_{session_number-1}.pth"
            model.load_state_dict(torch.load(model_path))
        new_model_path = f"../data/model/{method}_session_{session_number}.pth"
        model.train()

        loss_values = session_train(inputs, model, num_epochs)
        torch.save(model.state_dict(), new_model_path)
        if method == "our":
            buffer.replace()
        buffer.save(output_dir)


def evaluate(sesison_count=1):
    evaluate_by_cosine("er", sesison_count)
