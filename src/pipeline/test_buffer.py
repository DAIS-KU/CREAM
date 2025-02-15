import torch
from buffer import (
    Buffer,
    DataArguments,
    TevatronTrainingArguments,
    DenseModel,
    ModelArguments,
)
from transformers import BertTokenizer, BertModel
from collections import defaultdict
from typing import List

query_data = "../data/test_buffer_query.jsonl"
doc_data = "../data/test_buffer_doc.jsonl"
buffer_data = "../data"
qid_lst, docids_lst = ["5", "6", "7", "8"], ["15", "16", "17", "18"]
output_dir = "../data"

# https://github.com/caiyinqiong/L-2R/blob/main/src/tevatron/trainer.py
device = torch.device("mps")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def generate_test_buffer(method):
    import pickle

    buffer_qid2dids = defaultdict(
        list, {"5": ["15"], "6": ["16"], "7": ["17"], "8": ["18"]}
    )
    n_seen_so_far = defaultdict(int, {"5": 1, "6": 1, "7": 1, "8": 1})
    if "gss" in method:
        buffer_score = 0.5
        with open("../data/buffer.pkl", "wb") as output:
            pickle.dump((n_seen_so_far, buffer_qid2dids, buffer_score), output)
    else:
        with open("../data/buffer.pkl", "wb") as output:
            pickle.dump((n_seen_so_far, buffer_qid2dids), output)


# https://github.com/caiyinqiong/L-2R/blob/022f1282cd5bdf42c29d0f006e5c1b77c2c5c724/src/tevatron/data.py#L33C1-L42C20
# https://github.com/caiyinqiong/L-2R/blob/022f1282cd5bdf42c29d0f006e5c1b77c2c5c724/src/tevatron/data.py#L190
# https://github.com/caiyinqiong/L-2R/blob/022f1282cd5bdf42c29d0f006e5c1b77c2c5c724/src/tevatron/driver/train.py#L79
# https://github.com/caiyinqiong/L-2R/blob/main/src/tevatron/trainer.py
# _prepare_inputs <- inputs[2:] <- dataloader <- dataset
# qry_id, psg_ids, encoded_query, encoded_passages
# qid_lst, docids_lst = inputs[0], inputs[1]
# prepared = []
#         for x in inputs[2:]:
#             for key, val in x.items():
#                 x[key] = val.to(self.args.device)
#             prepared.append(x)
# TrainDataset(단일샘플, pad x, attention mask x), QPCollator(pad, attention mask 추가), DataLoader(batch size씩 iter) 어떻게 사용할지
def generate_test_qlst_dlst():
    max_q_len: int = 32
    max_p_len: int = 128

    def create_one_example(text_encoding: List[int], is_query=False):
        item = tokenizer.encode_plus(
            text_encoding,
            truncation="only_first",
            max_length=max_q_len if is_query else max_p_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        # print(f"item: {item}")# input_ids만 포함된 딕셔너리.
        return item

    def collate(features):
        qq_id = [f[0] for f in features]
        dd_id = [f[1] for f in features]
        qq = [f[2] for f in features]
        dd = [f[3] for f in features]

        if isinstance(qq_id[0], list):
            qq_id = sum(qq_id, [])
        if isinstance(dd_id[0], list):
            dd_id = sum(dd_id, [])
        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        # print(f"qq_id: {qq_id}")
        # print(f"dd_id: {dd_id}")
        # print(f"qq: {qq}")
        # print(f"dd: {dd}")
        q_collated = tokenizer.pad(
            qq,
            padding="max_length",
            max_length=max_q_len,
            return_tensors="pt",
        )
        d_collated = tokenizer.pad(
            dd,
            padding="max_length",
            max_length=max_p_len,
            return_tensors="pt",
        )
        # print(f"q_collated: {q_collated}")
        # print(f"d_collated: {d_collated}")
        return qq_id, dd_id, q_collated, d_collated

    qry_id = "5"
    qry = "query"
    encoded_query = [create_one_example(qry, is_query=True)]

    psg_ids = []
    encoded_passages = []
    psg_ids.append("15")
    encoded_passages.append(create_one_example("pos_psg"))
    psg_ids.append("16")
    encoded_passages.append(create_one_example("neg_psg"))

    features = list(zip(qry_id, psg_ids, encoded_query, encoded_passages))
    qq_id, dd_id, q_collated, d_collated = collate(features=features)
    return qq_id, dd_id, q_collated, d_collated


def random_retrieve_reservoir_update(method):
    generate_test_buffer(method)
    inputs = generate_test_qlst_dlst()
    qid_lst, docids_lst = inputs[0], inputs[1]
    prepared = []
    for x in inputs[2:]:
        for key, val in x.items():
            x[key] = val.to(device)
        prepared.append(x)
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    buffer = Buffer(
        model,
        tokenizer,
        DataArguments(
            query_data=query_data,
            doc_data=doc_data,
            buffer_data=buffer_data,
            mem_batch_size=3,
        ),
        TevatronTrainingArguments(output_dir=output_dir),
    )
    retrieval_result = buffer.retrieve(
        qid_lst=qid_lst,
        docids_lst=docids_lst,
    )
    print(f"random_retrieve_reservoir_update retrieval_result: {retrieval_result}")
    buffer.save(output_dir)

    update_result = buffer.update(
        qid_lst=qid_lst,
        docids_lst=docids_lst,
    )  # list: [num_q, 각 q에 대해 교체된 버퍼의 인덱스]
    print(f"random_retrieve_reservoir_update update_result: {update_result}")


def mir_retrieve_reservoir_update(method):
    generate_test_buffer(method)
    inputs = generate_test_qlst_dlst()
    qid_lst, docids_lst = inputs[0], inputs[1]
    prepared = []
    for x in inputs[2:]:
        for key, val in x.items():
            x[key] = val.to(device)
        prepared.append(x)
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    buffer = Buffer(
        model,
        tokenizer,
        DataArguments(
            retrieve_method="mir",
            query_data=query_data,
            doc_data=doc_data,
            buffer_data=buffer_data,
            mem_batch_size=3,
        ),
        TevatronTrainingArguments(output_dir=output_dir),
    )
    retrieval_result = buffer.retrieve(
        qid_lst=qid_lst, docids_lst=docids_lst, q_lst=prepared[0], d_lst=prepared[1]
    )
    print(f"mir_retrieve_reservoir_update retrieval_result: {retrieval_result}")
    buffer.save(output_dir)

    update_result = buffer.update(qid_lst=qid_lst, docids_lst=docids_lst)
    print(f"mir_retrieve_reservoir_update update_result: {update_result}")


def random_retrieve_gss_greedy_retrieve_update(method):
    generate_test_buffer(method)
    inputs = generate_test_qlst_dlst()
    qid_lst, docids_lst = inputs[0], inputs[1]
    prepared = []
    for x in inputs[2:]:
        for key, val in x.items():
            x[key] = val.to(device)
        prepared.append(x)
    # bert 외 정의한 dense model 사용
    model_args = ModelArguments(model_name_or_path="bert-base-uncased")
    training_args = TevatronTrainingArguments(output_dir=output_dir)
    print(f"training_args.device: {training_args.device}")
    model = DenseModel.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
    )
    model = model.to(
        "mps"
    )  # TrainingArguments.device가 자동으로 내부에서 mps로 설정함.
    model.train()  # gradient 계산을 위해 train mode로 설정
    buffer = Buffer(
        model,
        tokenizer,
        DataArguments(
            update_method="gss",
            query_data=query_data,
            doc_data=doc_data,
            buffer_data=buffer_data,
            mem_batch_size=1,
            gss_mem_strength=1,
            gss_batch_size=1,
        ),
        training_args,
    )
    retrieval_result = buffer.retrieve(
        qid_lst=qid_lst, docids_lst=docids_lst, q_lst=prepared[0], d_lst=prepared[1]
    )
    # print(f"retrieval_result: {retrieval_result}")
    buffer.save(output_dir)

    update_result = buffer.update(
        qid_lst=qid_lst, docids_lst=docids_lst, q_lst=prepared[0], d_lst=prepared[1]
    )
    print(f"update_result: {update_result}")


def l2r_retrieve_l2r_update(method):
    generate_test_buffer(method)
    inputs = generate_test_qlst_dlst()
    qid_lst, docids_lst = inputs[0], inputs[1]
    prepared = []
    for x in inputs[2:]:
        for key, val in x.items():
            x[key] = val.to(device)
        prepared.append(x)
    # bert 외 정의한 dense model 사용
    model_args = ModelArguments(model_name_or_path="bert-base-uncased")
    training_args = TevatronTrainingArguments(output_dir=output_dir)
    model = DenseModel.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
    )
    model = model.to(device)
    buffer = Buffer(
        model,
        tokenizer,
        DataArguments(
            retrieve_method="our",
            update_method="our",
            query_data=query_data,
            doc_data=doc_data,
            buffer_data=buffer_data,
            alpha=1.0,
            beta=1.0,
            mem_batch_size=1,
            new_batch_size=1,
        ),
        TevatronTrainingArguments(output_dir=output_dir),
    )
    # compatible mem_emb, mem_passage, pos_docids, candidate_neg_docids
    mem_passage, pos_docids, candidate_neg_docids = buffer.retrieve(
        qid_lst=qid_lst, docids_lst=docids_lst, q_lst=prepared[0], d_lst=prepared[1]
    )
    # print(f"retrieval_result: {retrieval_result}")
    buffer.save(output_dir)

    update_result = buffer.update(
        qid_lst=qid_lst,
        docids_lst=docids_lst,
        pos_docids=pos_docids,
        candidate_neg_docids=candidate_neg_docids,
    )
    print(f"update_result: {update_result}")


def test_buffer(method):
    if method == "random_retrieve_reservoir_update":
        random_retrieve_reservoir_update(method)
    elif method == "mir_retrieve_reservoir_update":
        mir_retrieve_reservoir_update(method)
    elif method == "random_retrieve_gss_greedy_retrieve_update":
        random_retrieve_gss_greedy_retrieve_update(method)
    elif method == "l2r_retrieve_l2r_update":
        l2r_retrieve_l2r_update(method)
    else:
        raise ValueError(f"Invalid buffer method {method}")
