import random
import torch
from transformers import BertTokenizer
from functions import (
    SimpleContrastiveLoss,
    evaluate_dataset,
    get_top_k_documents_by_cosine,
)

from data import (
    read_jsonl,
    write_file,
    renew_data,
    prepare_inputs,
    renew_data_mean_pooling,
)
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


def build_mir_buffer():
    query_data = f"/mnt/DAIS_NAS/huijeong/train_session0_queries.jsonl"
    doc_data = f"/mnt/DAIS_NAS/huijeong/train_session0_docs.jsonl"
    # buffer_data = "../data"
    output_dir = "../data"

    method = "l2r"
    model = build_model()
    # model_path = f"../data/model/{method}_session_0.pth"
    # model.load_state_dict(torch.load(model_path, weights_only=True))
    buffer = Buffer(
        model,
        tokenizer,
        DataArguments(
            retrieve_method="our",
            update_method="our",
            query_data=query_data,
            doc_data=doc_data,
            # buffer_data=buffer_data,
            alpha=1.0,
            beta=1.0,
            mem_batch_size=3,
            new_batch_size=3,
        ),
        TevatronTrainingArguments(output_dir=output_dir),
    )
    return buffer


# https://github.com/caiyinqiong/L-2R/blob/main/src/tevatron/trainer.py
def session_train(inputs, model, num_epochs, batch_size=8):
    # inputs : (q_lst, d_lst) = ( {q의 'input_ids', 'attention_mask'}, {docs의 'input_ids', 'attention_mask'})이 튜플이 원소인 2중리스트
    input_cnt = len(inputs)
    print(f"Total inputs #{input_cnt}")
    random.shuffle(inputs)

    loss_values = []
    loss_fn = SimpleContrastiveLoss()
    learning_rate = 2e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    batch_cnt = 0

    for epoch in range(num_epochs):
        total_loss, total_sec, batch_cnt = 0, 0, 0

        start_time = time.time()
        for start_idx in range(0, input_cnt, batch_size):
            end_idx = min(start_idx + batch_size, input_cnt)
            print(f"batch {start_idx}-{end_idx}")
            qreps_batch, dreps_batch = [], []
            for qid in range(start_idx, end_idx):
                q_tensors, docs_tensors = inputs[qid]
                output = model(q_tensors, docs_tensors)
                # output.q_reps: torch.Size([1, 768]), output.p_reps: torch.Size([8, 768])
                qreps_batch.append(output.q_reps)
                dreps_batch.append(output.p_reps)
                # print(f"output.q_reps:{output.q_reps.shape}, output.p_reps:{output.p_reps.shape}")
            q_embs, d_embs = torch.cat(qreps_batch, dim=0), torch.cat(
                dreps_batch, dim=0
            )
            # print(f"q_embs:{q_embs.shape}, d_embs:{d_embs.shape}")
            loss = loss_fn(q_embs, d_embs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss_values.append(loss.item())
            batch_cnt += 1
            print(
                f"Processed {qid}/{input_cnt} batches | Batch Loss: {loss.item():.4f} | Total Loss: {total_loss/batch_cnt:.4f}"
            )
        end_time = time.time()
        execution_time = end_time - start_time
        total_sec += execution_time
        print(
            f"Epoch {epoch} | Total {total_sec} seconds, Avg {total_sec / batch_cnt} seconds."
        )
    return loss_values


def train(session_count=4, num_epochs=1):
    buffer = build_mir_buffer()
    method = "l2r"
    output_dir = "../data"
    for session_number in range(session_count):
        print(f"Train Session {session_number}")
        query_path = (
            f"/mnt/DAIS_NAS/huijeong/train_session{session_number}_queries.jsonl"
        )
        doc_path = f"/mnt/DAIS_NAS/huijeong/train_session{session_number}_docs.jsonl"
        inputs = prepare_inputs(query_path, doc_path, buffer, method)

        model = build_model()
        model.to(devices[0])
        if session_number != 0:
            model_path = f"../data/model/{method}_session_{session_number-1}.pth"
            model.load_state_dict(torch.load(model_path, weights_only=True))
        new_model_path = f"../data/model/{method}_session_{session_number}.pth"
        model.train()

        loss_values = session_train(inputs, model, num_epochs)
        torch.save(model.state_dict(), new_model_path)
        if method == "l2r":
            buffer.replace()
        buffer.save(output_dir)


def evaluate(sesison_count=4):
    method = "l2r"
    for session_number in range(sesison_count):
        print(f"Evaluate Session {session_number}")
        eval_query_path = (
            f"/mnt/DAIS_NAS/huijeong/test_session{session_number}_queries.jsonl"
        )
        eval_doc_path = (
            f"/mnt/DAIS_NAS/huijeong/test_session{session_number}_docs.jsonl"
        )

        eval_query_data = read_jsonl(eval_query_path)
        eval_doc_data = read_jsonl(eval_doc_path)

        eval_query_count = len(eval_query_data)
        eval_doc_count = len(eval_doc_data)
        print(f"Query count:{eval_query_count}, Document count:{eval_doc_count}")

        rankings_path = f"../data/rankings/{method}_session_{session_number}.txt"
        model_path = f"../data/model/{method}_session_{session_number}.pth"

        start_time = time.time()
        new_q_data, new_d_data = renew_data_mean_pooling(
            queries_data=eval_query_data,
            documents_data=eval_doc_data,
            model_path=model_path,
        )
        end_time = time.time()
        print(f"Spend {end_time-start_time} seconds for encoding.")

        start_time = time.time()
        result = get_top_k_documents_by_cosine(new_q_data, new_d_data, k=10)
        end_time = time.time()
        print(f"Spend {end_time-start_time} seconds for retrieval.")

        rankings_path = f"../data/rankings/{method}_session_{session_number}.txt"
        write_file(rankings_path, result)
        evaluate_dataset(eval_query_path, rankings_path, eval_doc_count)
        del new_q_data, new_d_data
