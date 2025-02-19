import random
import torch
from transformers import BertTokenizer
from functions import (
    SimpleContrastiveLoss,
    evaluate_dataset,
    get_top_k_documents,
    write_file,
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


def get_top_k_documents_by_cosine(query_emb, current_data_embs, k=10):
    scores = torch.nn.functional.cosine_similarity(
        query_emb.unsqueeze(0), current_data_embs, dim=1
    )
    top_k_scores, top_k_indices = torch.topk(scores, k)
    return top_k_indices.tolist()


# https://github.com/caiyinqiong/L-2R/blob/main/src/tevatron/trainer.py
def session_train(inputs, model, num_epochs):
    input_cnt = len(
        inputs
    )  # (q_lst, d_lst, identity, old_emb) List[Dict[str, Union[torch.Tensor, Any]]]
    loss_values = []

    loss_fn = SimpleContrastiveLoss()
    learning_rate = 2e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    batch_size = 32

    for epoch in range(num_epochs):
        total_loss = 0

        for start_idx in range(0, input_cnt, batch_size):
            end_idx = min(start_idx + batch_size, input_cnt)
            print(f"batch {start_idx}-{end_idx}")

            input_batch = []
            for _id in range(start_idx, end_idx):
                input_embeddings = model(inputs[_id])
                input_batch.append(input_embeddings)
            print(f"input_batch: {input_embeddings.shape}")
            # EncoderOutput(loss=loss, scores=scores, q_reps=q_reps, p_reps=p_reps)의 loss를 사용해도 되는건가???
            # SimpleContrastiveLoss(x: Tensor, y: Tensor, ...)이면 loss_fn(x=q_reps, y=p_reps)로 사용해야 하는건가???
            loss = loss_fn(input_batch)

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


def train(method, session_count=1, num_epochs=1):
    buffer = Buffer(model, tokenizer, DataArguments(), TevatronTrainingArguments())
    for session_number in range(session_count):
        print(f"Train Session {session_number}")
        query_path = f"../data/sessions/train_session{session_number}_queries.jsonl"
        doc_path = f"../data/sessions/train_session{session_number}_docs.jsonl"
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

        loss_values = session_train(inputs, num_epochs, method, buffer, session_number)
        torch.save(model.state_dict(), new_model_path)
        if method == "our":
            buffer.replace()
        buffer.save(output_dir)


def evaluate(method, sesison_count=1):
    for session_number in range(sesison_count):
        print(f"Evaluate Session {session_number}")
        eval_query_path = f"../data/sessions/test_session{session_number}_queries.jsonl"
        eval_doc_path = f"../data/sessions/test_session{session_number}_docs.jsonl"

        eval_query_data = read_jsonl(eval_query_path)[:10]
        eval_doc_data = read_jsonl(eval_doc_path)[:100]

        eval_query_count = len(eval_query_data)
        eval_doc_count = len(eval_doc_data)
        print(f"Query count:{eval_query_count}, Document count:{eval_doc_count}")

        rankings_path = f"../data/rankings/{method}_session_{session_number}.txt"
        model_path = f"../data/model/{method}_session_{session_number}.pth"

        start_time = time.time()
        new_q_data, new_d_data = renew_data(
            queries=eval_query_data,
            documents=eval_doc_data,
            nbits=0,
            embedding_dim=768,
            model_path=model_path,
            renew_q=True,
            renew_d=True,
        )
        end_time = time.time()
        print(f"Spend {end_time-start_time} seconds for encoding.")

        start_time = time.time()
        result = get_top_k_documents(new_q_data, new_d_data, k=10)
        end_time = time.time()
        print(f"Spend {end_time-start_time} seconds for retrieval.")

        rankings_path = f"../data/rankings/{method}_session_{session_number}.txt"
        write_file(rankings_path, result)
        evaluate_dataset(eval_query_path, rankings_path, eval_doc_count)
        del new_q_data, new_d_data
