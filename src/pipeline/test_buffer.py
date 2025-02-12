import torch
from buffer import Buffer, DataArguments, TevatronTrainingArguments
from transformers import BertTokenizer, BertModel
from collections import defaultdict


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = BertModel.from_pretrained("bert-base-uncased").to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# https://github.com/caiyinqiong/L-2R/blob/main/src/tevatron/trainer.py
def save(self, output_dir):
    if self.data_args.cl_method:
        self.buffer.save(output_dir)


def generate_test_buffer(method):
    import pickle

    buffer_qid2dids = defaultdict(list, {5: [15], 6: [16], 7: [17], 8: [18]})
    n_seen_so_far = 4
    if "gss" in method:
        buffer_score = 0.5
        with open("../data/buffer.pkl", "wb") as output:
            pickle.dump((n_seen_so_far, buffer_qid2dids, buffer_score), output)
    else:
        with open("../data/buffer.pkl", "wb") as output:
            pickle.dump((n_seen_so_far, buffer_qid2dids), output)


# https://github.com/caiyinqiong/L-2R/blob/main/src/tevatron/trainer.py
# https://github.com/caiyinqiong/L-2R/blob/main/src/tevatron/data.py#L18
# _prepare_inputs <- inputs[2:] <- dataloader <- dataset
# qry_id, psg_ids, encoded_query, encoded_passages
# qid_lst, docids_lst = inputs[0], inputs[1]
# prepared = []
#         for x in inputs[2:]:
#             for key, val in x.items():
#                 x[key] = val.to(self.args.device)
#             prepared.append(x)
# https://github.com/caiyinqiong/L-2R/blob/022f1282cd5bdf42c29d0f006e5c1b77c2c5c724/src/tevatron/data.py#L33C1-L42C20
def generate_test_qlst_dlst():
    return None, None


def test_buffer(method):
    generate_test_buffer(method)
    query_data = "../data/test_buffer_query.jsonl"
    doc_data = "../data/test_buffer_doc.jsonl"
    buffer_data = "../data"
    output_dir = "../data"
    kwargs = {}

    if method == "random_retrieve_reservoir_update":
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
    elif method == "mir_retrieve_reservoir_update":
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
    elif method == "random_retrieve_gss_greedy_retrieve_update":
        buffer = Buffer(
            model,
            tokenizer,
            DataArguments(
                update_method="gss",
                query_data=query_data,
                doc_data=doc_data,
                buffer_data=buffer_data,
                mem_batch_size=3,
                gss_batch_size=3,
            ),
            TevatronTrainingArguments(output_dir=output_dir),
        )
        q_lst, d_lst = generate_test_qlst_dlst()
        kwargs = {
            "q_lst": q_lst,
            "d_lst": d_lst,
        }
    elif method == "l2r_retrieve_l2r_update":
        buffer = Buffer(
            model,
            tokenizer,
            DataArguments(
                retrieve_method="our",
                update_method="our",
                query_data=query_data,
                doc_data=doc_data,
                buffer_data=buffer_data,
                mem_batch_size=3,
            ),
            TevatronTrainingArguments(output_dir=output_dir),
        )
        q_lst, d_lst = generate_test_qlst_dlst()
        kwargs = {
            "q_lst": q_lst,
            "d_lst": d_lst,
        }
    else:
        raise ValueError(f"Invalid buffer method {method}")

    qid_lst, docids_lst = [5, 6, 7, 8], [15, 16, 17, 18]
    retrieval_result = buffer.retrieve(qid_lst=qid_lst, docids_lst=docids_lst, **kwargs)
    print(f"retrieval_result: {retrieval_result}")
    buffer.save(output_dir)

    update_result = buffer.update(qid_lst=qid_lst, docids_lst=docids_lst, **kwargs)
    print(f"update_result: {update_result}")
