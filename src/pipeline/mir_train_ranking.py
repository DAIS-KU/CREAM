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
        model.load_state_dict(torch.load(model_path))
    return model


def build_er_buffer():
    query_data = f"/mnt/DAIS_NAS/huijeong/train_session0_queries.jsonl"
    doc_data = f"/mnt/DAIS_NAS/huijeong/train_session0_docs.jsonl"
    # buffer_data = "../data"
    output_dir = "../data"

    method = "er"
    model = build_model()
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


# https://github.com/caiyinqiong/L-2R/blob/main/src/tevatron/trainer.py
def session_train(inputs, model, num_epochs):
    # inputs : (q_lst, d_lst) = ([qid], [doc_ids], {q의 'input_ids', 'attention_mask'}, {docs의 'input_ids', 'attention_mask'})이 튜플이 원소인 리스트
    # 이미 배치 처리가 되어있어서, inputs의 각 원소만 쌓아서 연산하면됨
    input_cnt = 1  # len(inputs)
    print(f"input_cnt: {input_cnt}")
    # (q_lst, d_lst, identity, old_emb) List[Dict[str, Union[torch.Tensor, Any]]]
    # q_lst = ([qid], [p1n7 doc_id], {'input_ids': ...}, {'attention_mask': ...})
    # q_lst[2] torch.Size([1, 32]), q_lst[3] torch.Size([8, 128])

    loss_values = []
    loss_fn = SimpleContrastiveLoss()
    learning_rate = 2e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_cnt = 0

    for epoch in range(num_epochs):
        total_loss = 0

        for _id in range(0, input_cnt):
            for _iid in range(0, len(inputs[_id])):
                # to(model.next(model.parameters()).device)
                output = model(inputs[_id][_iid][2], inputs[_id][_iid][3])
                # output.q_reps: torch.Size([1, 768]), output.p_reps: torch.Size([8, 768])
                loss = loss_fn(output.q_reps, output.p_reps)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                loss_values.append(loss.item())
                loss_cnt += 1
                print(
                    f"Processed ({_id}), {_iid}/{len(inputs[_id])} batches | Batch Loss: {loss.item():.4f} | Total Loss: {total_loss/loss_cnt:.4f}"
                )
    return loss_values


def train(session_count=1, num_epochs=1):
    buffer = build_er_buffer()
    method = "er"
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


def evaluate_by_cosine(method, sesison_count=1):
    for session_number in range(sesison_count):
        print(f"Evaluate Session {session_number}")
        eval_query_path = (
            f"/mnt/DAIS_NAS/huijeong/test_session{session_number}_queries.jsonl"
        )
        eval_doc_path = (
            f"/mnt/DAIS_NAS/huijeong/test_session{session_number}_docs.jsonl"
        )

        eval_query_data = read_jsonl(eval_query_path)[:10]
        eval_doc_data = read_jsonl(eval_doc_path)[:100]

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
