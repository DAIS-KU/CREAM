import time
import random
import torch
from transformers import BertTokenizer
import os

from functions import (
    SimpleContrastiveLoss,
    evaluate_dataset,
    get_top_k_documents_by_cosine,
    renew_data_mean_pooling,
)

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
    write_line,
    write_file,
    read_jsonl_as_dict,
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
        bert_state_dict = torch.load(bert_weight_path, map_location=devices[-1])
        model.lm_q.load_state_dict(bert_state_dict)
        model.lm_p.load_state_dict(bert_state_dict)

    if model_path:
        model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location=devices[-1])
        )
    # model.to(devices[0])
    return model


def build_gss_buffer(new_batch_size, mem_batch_size, compatible, buffer_path=None):
    # query_data = f"/home/work/retrieval/data/datasetM_large/train_session0_queries_cos.jsonl"
    query_data = (
        f"/home/work/retrieval/data/datasetM_large/train_session0_queries.jsonl"
    )
    doc_data = f"/home/work/retrieval/data/datasetM_large/train_session0_docs.jsonl"
    # doc_data = f"/mnt/DAIS_NAS/huijeong/sub/train_session{session_number}_docs.jsonl"
    buffer_data = "./data"  # comp시에는 필요
    output_dir = "/home/work/retrieval/data/gss_output"

    method = "gss"
    # model = build_model(bert_weight_path="/mnt/DAIS_NAS/huijeong/model/base_model_lotte.pth")
    model = build_model()

    buffer = Buffer(
        model,
        tokenizer,
        DataArguments(
            # --------------------수정 여부 판단--------------------
            retrieve_method="random",
            update_method="gss",
            query_data=query_data,
            doc_data=None,  # 수정
            # buffer_data=buffer_data,
            mem_batch_size=mem_batch_size,
            new_batch_size=new_batch_size,
            compatible=compatible,
            mem_size=30,
            gss_batch_size=3,  # 메모리에서 하나의 배치 샘플링시 포함할 샘플 개수
            gss_mem_strength=3,  # 메모리에서 최대 몇 개의 배치를 샘플링할 지
            # --------------------수정 여부 판단--------------------
        ),
        TevatronTrainingArguments(output_dir=output_dir),
    )

    if buffer_path and os.path.exists(buffer_path):
        buffer.load(buffer_path)

    return buffer


# https://github.com/caiyinqiong/L-2R/blob/main/src/tevatron/trainer.py
def session_train(
    session_number, inputs, model, buffer, num_epochs, batch_size=8, compatible=False
):
    # inputs : (q_lst, d_lst) = ( {q의 'input_ids', 'attention_mask'}, {docs의 'input_ids', 'attention_mask'})이 튜플이 원소인 2중리스트
    input_cnt = len(inputs)
    print(f"Total inputs #{input_cnt} with compatiblity {compatible}")
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
                if not compatible or session_number == 0:
                    q_tensors, docs_tensors, docid_lst = inputs[qid]
                    output = model(q_tensors, docs_tensors)
                else:
                    q_tensors, docs_tensors, identity, doc_oldemb, docid_lst = inputs[
                        qid
                    ]
                    output = model(q_tensors, docs_tensors, identity, doc_oldemb)
                # output.q_reps: torch.Size([1, 768]), output.p_reps: torch.Size([8, 768])
                qreps_batch.append(output.q_reps)
                dreps_batch.append(output.p_reps)
                # print(f"output.q_reps:{output.q_reps.shape}, output.p_reps:{output.p_reps.shape}")
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
    # resume_session=1,  # 첫 학습이면 0으로 수정
):

    method = "gss"
    output_dir = "/home/work/retrieval/data/gss_output"
    buffer_path = None
    # buffer_path = f"{output_dir}/buffer.pkl" if resume_session > 0 else None
    buffer = build_gss_buffer(new_batch_size, mem_batch_size, compatible, buffer_path)
    start_time = time.time()
    time_values_path = f"../data/loss/total_time_gss_datasetM_large.txt"
    for session_number in range(session_count):
        print(f"Train Session {session_number}")
        doc_path = f"/home/work/retrieval/data/datasetM_large/train_session{session_number}_docs.jsonl"
        query_path = (
            f"/home/work/retrieval/data/datasetM_large/train_session0_queries.jsonl"
        )
        # query_path = f"/home/work/retrieval/data/datasetM_large/train_session0_queries_cos.jsonl"

        # doc_path = f"/mnt/DAIS_NAS/huijeong/sub/train_session{session_number}_docs.jsonl"

        # if session_number < 3: # 수정
        #     # query_path = f"/mnt/DAIS_NAS/huijeong/sub/train_session{session_number}_queries.jsonl" # 수정
        #     query_path = f"/home/sunho/continual_retrieval/ContinualRetrieval_paper/train_session{session_number}_queries_cos.jsonl"
        # else:

        #     # query_path = f"/mnt/DAIS_NAS/huijeong/train_session0_queries.jsonl" # 수정
        #     query_path = f"/home/sunho/continual_retrieval/ContinualRetrieval_paper/train_session(0,1,2)_queries_cos.jsonl"

        # query_path = f"/mnt/DAIS_NAS/huijeong/sub/train_session{session_number}_queries.jsonl"
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

        # model = build_model(bert_weight_path="/mnt/DAIS_NAS/huijeong/model/base_model_lotte.pth")
        model = build_model()
        # model.to(device)

        if session_number != 0:
            model_path = f"/home/work/retrieval/data/model/{method}MS_session_{session_number-1}.pth"
            print(f"Load model {model_path}")
            model.load_state_dict(
                torch.load(model_path, weights_only=True, map_location=device)
            )
        new_model_path = (
            f"/home/work/retrieval/data/model/{method}MS_session_{session_number}.pth"
        )
        model.train()

        loss_values = session_train(
            session_number, inputs, model, buffer, num_epochs, batch_size, compatible
        )
        torch.save(model.state_dict(), new_model_path)
        buffer.save(output_dir)
    end_time = time.time()
    total_sec = end_time - start_time
    print(f"{total_sec} sec. ")
    write_line(time_values_path, f"({total_sec}sec)\n", "a")


# 추가한 부분 : model_builder 함수
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
    method = "gss"
    for session_number in range(sesison_count):
        print(f"Evaluate Session {session_number}")

        if session_number != 0:
            continue
        eval_query_path = (
            f"/home/work/retrieval/data/datasetM_large/test_session{session_number}_queries.jsonl"
            # f"/mnt/DAIS_NAS/huijeong/sub/test_session{session_number}_queries.jsonl"
        )
        eval_doc_path = (
            f"/home/work/retrieval/data/datasetM_large/test_session{session_number}_docs.jsonl"
            # f"/mnt/DAIS_NAS/huijeong/sub/test_session{session_number}_docs.jsonl"
        )
        eval_query_data = read_jsonl(eval_query_path, True)
        eval_doc_data = read_jsonl(eval_doc_path, False)

        eval_query_count = len(eval_query_data)
        eval_doc_count = len(eval_doc_data)
        print(f"Query count:{eval_query_count}, Document count:{eval_doc_count}")

        rankings_path = f"/home/work/retrieval/data/rankings/{method}MS_session_{session_number}.txt"
        model_path = f"../data/model/{method}MS_session_{session_number}.pth"

        start_time = time.time()

        # 추가한 부분 : 리스트에서 딕셔너리로 변환
        eval_query_data, eval_doc_data = renew_data_mean_pooling(
            model_builder, model_path, eval_query_data, eval_doc_data
        )

        result = get_top_k_documents_by_cosine(eval_query_data, eval_doc_data, 10)
        end_time = time.time()
        print(f"Spend {end_time-start_time} seconds for retrieval.")

        rankings_path = f"/home/work/retrieval/data/rankings/{method}MS_session_{session_number}.txt"
        write_file(rankings_path, result)
        eval_log_path = (
            f"/home/work/retrieval/data/evals/{method}MS_{session_number}.txt"
        )
        evaluate_dataset(eval_query_path, rankings_path, eval_doc_count, eval_log_path)
