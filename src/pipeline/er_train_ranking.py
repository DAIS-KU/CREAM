import random
import time

import torch
from transformers import BertTokenizer

from buffer import (
    Buffer,
    DataArguments,
    DenseModel,
    ModelArguments,
    TevatronTrainingArguments,
)
from data import (
    load_eval_docs,
    prepare_inputs,
    read_jsonl,
    write_file,
    write_line,
    read_jsonl_as_dict,
)

from functions import (
    SimpleContrastiveLoss,
    evaluate_dataset,
    get_top_k_documents_by_cosine,
    renew_data_mean_pooling,
)
import time


torch.autograd.set_detect_anomaly(True)
tokenizer = BertTokenizer.from_pretrained(
    "/home/work/retrieval/bert-base-uncased/bert-base-uncased"
)

num_gpus = torch.cuda.device_count()
devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
device = devices[-1] if num_gpus > 0 else torch.device("cpu")


def build_model(bert_weight_path=None, model_path=None):
    model_args = ModelArguments(
        model_name_or_path="/home/work/retrieval/bert-base-uncased/bert-base-uncased"
    )
    training_args = TevatronTrainingArguments(
        output_dir="/home/work/retrieval/data/model"
    )
    model = DenseModel.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
    )
    if bert_weight_path:
        bert_state_dict = torch.load(bert_weight_path)
        model.lm_q.load_state_dict(bert_state_dict)
        model.lm_p.load_state_dict(bert_state_dict)

    if model_path:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    # model.to(devices[1])
    return model


def build_er_buffer(new_batch_size, mem_batch_size, compatible):
    # query_data = f"/home/work/retrieval/data/datasetL_large/train_session0_queries_cos.jsonl"
    query_data = (
        f"/home/work/retrieval/data/datasetM_large/train_session0_queries.jsonl"
    )
    doc_data = f"/home/work/retrieval/data/datasetM_large/train_session0_docs.jsonl"

    buffer_data = "../data"  # comp시에는 필요
    output_dir = "/home/work/retrieval/data/er_output"
    # bert_weight_path = "/mnt/DAIS_NAS/huijeong/model/base_model_lotte.pth" # .pth

    method = "er"
    model = build_model()

    # model_path = f"../data/model/{method}_session_0.pth"

    # model.load_state_dict(torch.load(model_path, weights_only=True))
    buffer = Buffer(
        model,
        tokenizer,
        DataArguments(
            query_data=query_data,
            doc_data=None,  # doc_data,
            # buffer_data=buffer_data,
            new_batch_size=new_batch_size,
            mem_batch_size=mem_batch_size,
            compatible=compatible,
            mem_size=30,
        ),
        TevatronTrainingArguments(output_dir=output_dir),
    )
    return buffer


# https://github.com/caiyinqiong/L-2R/blob/main/src/tevatron/trainer.py
def session_train(inputs, model, buffer, num_epochs, batch_size=32, compatible=False):
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
                q_tensors, docs_tensors, docid_lst = inputs[qid]
                output = model(q_tensors, docs_tensors)
                # output.q_reps: torch.Size([1, 768]), output.p_reps: torch.Size([8, 768])
                qreps_batch.append(output.q_reps)
                dreps_batch.append(output.p_reps)
                # print(
                #     f"output.q_reps:{output.q_reps.shape}, output.p_reps:{output.p_reps.shape}"
                # )
            if compatible:
                buffer.update_old_embs(docid_lst, output.p_reps)
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


def train(
    session_count=10,
    num_epochs=1,
    batch_size=32,
    compatible=False,
    new_batch_size=3,
    mem_batch_size=3,
):
    buffer = build_er_buffer(new_batch_size, mem_batch_size, compatible)
    method = "er"
    output_dir = "/home/work/retrieval/data/er_output"
    start_time = time.time()
    time_values_path = f"../data/loss/total_time_er_datasetM_large.txt"
    for session_number in range(session_count):
        print(f"Train Session {session_number}")
        # session0에 대한 쿼리로만 학습(문서만 바뀜)
        # query_path = f"/mnt/DAIS_NAS/huijeong/sub/train_session{session_number}_queries.jsonl"
        doc_path = f"/home/work/retrieval/data/datasetM_large/train_session{session_number}_docs.jsonl"
        query_path = (
            f"/home/work/retrieval/data/datasetM_large/train_session0_queries.jsonl"
        )
        # query_path = f"/home/work/retrieval/data/datasetL_large/train_session0_queries_cos.jsonl"
        inputs = prepare_inputs(
            session_number,
            query_path,
            doc_path,
            buffer,
            method,
            new_batch_size,
            mem_batch_size,
            compatible,
        )
        # bert_weight_path = "/mnt/DAIS_NAS/huijeong/model/base_model_lotte.pth"
        model = build_model()

        if session_number != 0:
            model_path = f"/home/work/retrieval/data/model/{method}LS_session_{session_number-1}.pth"
            print(f"Load model {model_path}")
            model.load_state_dict(torch.load(model_path, weights_only=True))
        new_model_path = (
            f"/home/work/retrieval/data/model/{method}LS_session_{session_number}.pth"
        )
        model.train()

        loss_values = session_train(
            inputs, model, buffer, num_epochs, batch_size, compatible
        )
        torch.save(model.state_dict(), new_model_path)
        buffer.save(output_dir)
    end_time = time.time()
    total_sec = end_time - start_time
    print(f"{total_sec} sec. ")
    write_line(time_values_path, f"({total_sec}sec)\n", "a")


def model_builder(model_path=None):
    model_args = ModelArguments(
        model_name_or_path="/home/work/retrieval/bert-base-uncased/bert-base-uncased"
    )
    training_args = TevatronTrainingArguments(output_dir="../data/model")
    model = DenseModel.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
    )
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=devices[-1]))
    # model.to(devices[0])
    return model


def evaluate(sesison_count=10):
    method = "er"
    for session_number in range(sesison_count):
        print(f"Evaluate Session {session_number}")
        eval_query_path = f"/home/work/retrieval/data/datasetM_large/test_session{session_number}_queries.jsonl"
        # eval_doc_path = f"/home/work/retrieval/data/datasetM_large/train_session{session_number}_docs.jsonl"
        eval_doc_path = f"/home/work/retrieval/data/datasetM_large/test_session{session_number}_docs.jsonl"
        eval_query_data = read_jsonl(eval_query_path, True)
        eval_doc_data = read_jsonl(eval_doc_path, False)

        eval_query_count = len(eval_query_data)
        eval_doc_count = len(eval_doc_data)
        print(f"Query count:{eval_query_count}, Document count:{eval_doc_count}")

        rankings_path = f"../data/rankings/{method}LS_session_{session_number}.txt"
        model_path = f"../data/model/{method}LS_session_{session_number}.pth"

        start_time = time.time()

        # 추가한 부분 : 리스트에서 딕셔너리로 변환
        eval_query_data, eval_doc_data = renew_data_mean_pooling(
            model_builder, model_path, eval_query_data, eval_doc_data
        )

        """
        # 인자 개수 수정
        result = get_top_k_documents_by_cosine(eval_query_data, eval_doc_data, 10, method, session_number)
        """
        result = get_top_k_documents_by_cosine(eval_query_data, eval_doc_data, 10)
        end_time = time.time()
        print(f"Spend {end_time-start_time} seconds for retrieval.")

        write_file(rankings_path, result)
        eval_log_path = f"../data/evals/{method}LS_{session_number}.txt"
        evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
