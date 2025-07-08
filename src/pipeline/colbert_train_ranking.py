import random
import time

import torch
from transformers import BertTokenizer, BertTokenizerFast

from buffer import (
    Buffer,
    DataArguments,
    ColbertModel,
    ModelArguments,
    TevatronTrainingArguments,
)
from data import (
    load_eval_docs,
    write_line,
    read_jsonl,
    write_file,
    prepare_colbert_inputs,
)
from functions import (
    SimpleContrastiveLoss,
    evaluate_dataset,
    colbert_renew_data,
    get_top_k_documents,
)

torch.autograd.set_detect_anomaly(True)
tokenizer = BertTokenizerFast.from_pretrained(
    "/home/work/retrieval/bert-base-uncased/bert-base-uncased"
)
# ColBERT 논문에 등장하는 [Q], [D] 두 토큰을 신규 등록
special_tokens_dict = {"additional_special_tokens": ["[Q]", "[D]"]}
num_new = tokenizer.add_special_tokens(special_tokens_dict)

num_gpus = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]


def build_model(bert_weight_path=None, model_path=None):
    model_args = ModelArguments(
        model_name_or_path="bert-base-uncased", add_pooler=True, projection_out_dim=128
    )
    training_args = TevatronTrainingArguments(output_dir="../data/model")
    model = ColbertModel.build(
        model_args,
        training_args,
    )
    if bert_weight_path:
        bert_state_dict = torch.load(bert_weight_path, map_location=devices[-1])
        model.lm_q.load_state_dict(bert_state_dict)
        model.lm_p.load_state_dict(bert_state_dict)

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=devices[-1]))
    model.to(devices[0])
    # model.resize_token_embeddings(len(tokenizer))
    return model


def session_train(session_number, model, num_epochs, batch_size=32):
    # inputs : (q_lst, d_lst) = ( {q의 'input_ids', 'attention_mask'}, {docs의 'input_ids', 'attention_mask'})이 튜플이 원소인 2중리스트
    loss_values = []
    loss_fn = SimpleContrastiveLoss()
    learning_rate = 5e-6
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    batch_cnt = 0

    for epoch in range(num_epochs):
        inputs = prepare_colbert_inputs(session_number)
        total_loss, total_sec, batch_cnt = 0, 0, 0
        input_cnt = len(inputs)
        random.shuffle(inputs)

        print(
            f"[session_number {session_number} / epoch {epoch}] Total inputs #{input_cnt}"
        )

        start_time = time.time()
        for start_idx in range(0, input_cnt, batch_size):
            end_idx = min(start_idx + batch_size, input_cnt)
            print(f"batch {start_idx}-{end_idx}")

            batch_q_input_ids, batch_q_attention = [], []
            batch_d_input_ids, batch_d_attention = [], []
            for qid in range(start_idx, end_idx):
                q_tensors, docs_tensors = inputs[qid]
                batch_q_input_ids.append(q_tensors["input_ids"])
                batch_q_attention.append(q_tensors["attention_mask"])
                batch_d_input_ids.append(docs_tensors["input_ids"])
                batch_d_attention.append(docs_tensors["attention_mask"])

            batch_q = {
                "input_ids": torch.cat(batch_q_input_ids, dim=0),
                "attention_mask": torch.cat(batch_q_attention, dim=0),
            }
            batch_docs = {
                "input_ids": torch.cat(batch_d_input_ids, dim=0),
                "attention_mask": torch.cat(batch_d_attention, dim=0),
            }

            device = next(model.parameters()).device
            batch_q = {k: v.to(device) for k, v in batch_q.items()}
            batch_docs = {k: v.to(device) for k, v in batch_docs.items()}

            output = model(query=batch_q, passage=batch_docs)
            loss = output.loss

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


def train(
    session_count=10,
    num_epochs=1,
    batch_size=32,
    compatible=False,
    new_batch_size=3,
    mem_batch_size=3,
    mem_upsample=6,
):
    method = "colbert"
    output_dir = "../data"
    total_sec = 0
    for session_number in range(session_count):
        start_time = time.time()
        time_values_path = (
            f"../data/loss/total_time_colbert_datasetL_large_share_{session_number}.txt"
        )
        print(f"Train Session {session_number}")
        query_path = f"/home/work/retrieval/data/datasetL_large_share/train_session{session_number}_queries.jsonl"
        doc_path = f"/home/work/retrieval/data/datasetL_large_share/train_session{session_number}_docs.jsonl"
        model = build_model()

        if session_number != 0:
            model_path = f"../data/model/{method}_session_{session_number-1}.pth"
            print(f"Load model {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=devices[-1]))
        new_model_path = f"../data/model/{method}_session_{session_number}.pth"
        model.train()

        loss_values = session_train(session_number, model, num_epochs, batch_size)
        torch.save(model.state_dict(), new_model_path)
        end_time = time.time()
        total_sec += end_time - start_time
        print(f"{end_time-start_time} sec. ")
    write_line(time_values_path, f"({total_sec}sec)\n", "a")


def model_builder(model_path=None):
    model_args = ModelArguments(
        model_name_or_path="bert-base-uncased", add_pooler=True, projection_out_dim=128
    )
    training_args = TevatronTrainingArguments(output_dir="../data/model")
    model = ColbertModel.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
    )
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=devices[0]))
    # model.to(devices[0])
    return model


def evaluate(session_count=10):
    for session_number in range(6, session_count):
        _evaluate(session_number)


def _evaluate(session_number):
    method = "colbert"
    print(f"Evaluate Session {session_number}")
    eval_query_path = (
        f"../data/datasetL_large_share/test_session{session_number}_queries.jsonl"
    )
    # eval_doc_path = f"../data/datasetL_large_share/test_session{session_number}_docs.jsonl"
    eval_doc_path = (
        f"../data/datasetL_large_share/train_session{session_number}_docs.jsonl"
    )

    eval_query_data = read_jsonl(eval_query_path, True)
    eval_doc_data = read_jsonl(eval_doc_path, False)

    eval_query_count = len(eval_query_data)
    eval_doc_count = len(eval_doc_data)
    print(f"Query count:{eval_query_count}, Document count:{eval_doc_count}")

    rankings_path = f"../data/rankings/{method}_session_{session_number}.txt"
    model_path = f"../data/model/{method}_session_{session_number}.pth"

    start_time = time.time()
    new_q_data, new_d_data = colbert_renew_data(
        queries=eval_query_data,
        documents=eval_doc_data,
        model_path=model_path,
        model_builder=model_builder,
        renew_q=True,
        renew_d=True,
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
