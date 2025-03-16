import collections
import json
import os
import pickle

import torch
from data import load_train_docs
from functions import convert_str_id_to_number_id

from .arguments import DataArguments, TevatronTrainingArguments
from .gss_greedy_update import GSSGreedyUpdate
from .l2r_retrieve import L2R_retrieve
from .l2r_update import L2RUpdate
from .mir_retrieve import MIR_retrieve
from .ocs_retrieve import OCS_retrieve
from .random_retrieve import Random_retrieve
from .reservoir_update import Reservoir_update

retrieve_methods = {
    "random": Random_retrieve,
    "mir": MIR_retrieve,
    "ocs": OCS_retrieve,
    "our": L2R_retrieve,
}
update_methods = {
    "random": Reservoir_update,
    "gss": GSSGreedyUpdate,
    "our": L2RUpdate,
}


class Buffer(torch.nn.Module):
    def __init__(
        self,
        model,
        tokenizer,
        params: DataArguments,
        train_params: TevatronTrainingArguments,
    ):
        super().__init__()
        self.params = params
        self.train_params = train_params
        self.model = model
        self.tokenizer = tokenizer
        self.buffer_size = params.mem_size
        self.n_seen_so_far = collections.defaultdict(
            int
        )  # 目前已经过了多少个样本了, 只有er需要
        self.buffer_qid2dids = collections.defaultdict(list)
        self.buffer_did2emb = collections.defaultdict(None)
        self.compatible = params.compatible

        if self.params.update_method == "gss":
            buffer_score = None

        # if params.buffer_data:
        #     print("load buffer data from %s" % params.buffer_data)
        #     pkl_file = open(os.path.join(params.buffer_data, "buffer.pkl"), "rb")
        #     if self.params.update_method == "gss":
        #         self.n_seen_so_far, self.buffer_qid2dids, buffer_score = pickle.load(
        #             pkl_file
        #         )
        #     else:
        #         self.n_seen_so_far, self.buffer_qid2dids = pickle.load(pkl_file)
        #     pkl_file.close()
        #     # print(
        #     #     f"Load n_seen_so_far: {self.n_seen_so_far}, buffer_qid2dids:{self.buffer_qid2dids}"
        #     # )

        #     if params.compatible:
        #         pkl_file = open(
        #             os.path.join(params.buffer_data, "buffer_emb.pkl"), "rb"
        #         )
        #         self.buffer_did2emb = pickle.load(pkl_file)
        #         pkl_file.close()

        self.qid2query = self.read_data(
            is_query=True, data_path=params.query_data
        )  # {'qid':query text}
        if params.doc_data == None:
            self.did2doc = load_train_docs()
            self.did2doc = {
                doc_id: value["text"] for doc_id, value in self.did2doc.items()
            }
        else:
            self.did2doc = self.read_data(
                is_query=False, data_path=params.doc_data
            )  # {'doc_id':doc text}
        print("total did2doc:", len(self.did2doc))

        if self.params.update_method == "gss":
            self.update_method = update_methods[params.update_method](
                params, train_params, buffer_score=buffer_score
            )
        else:
            self.update_method = update_methods[params.update_method](
                params, train_params
            )
        self.retrieve_method = retrieve_methods[params.retrieve_method](
            params, train_params
        )

    def init(self, qid, doc_ids):
        print(f"init buffer {qid} {doc_ids}")
        self.buffer_qid2dids[qid] = doc_ids
        self.n_seen_so_far[qid] = len(doc_ids)

    def read_data(self, is_query, data_path):
        print("load data from %s" % data_path)
        id2text = {}
        with open(data_path, "r") as f:
            for line in f:
                data = json.loads(line)
                if is_query:
                    if self.compatible:
                        data["qid"] = convert_str_id_to_number_id(data["qid"])
                    id2text[data["qid"]] = data["query"]
                else:
                    if self.compatible:
                        data["doc_id"] = convert_str_id_to_number_id(data["doc_id"])
                    id2text[data["doc_id"]] = (
                        data["title"]
                        + self.params.passage_field_separator
                        + data["text"]
                        if "title" in data
                        else data["text"]
                    )
        return id2text

    def update(self, qid_lst, docids_lst, **kwargs):
        return self.update_method.update(
            buffer=self, qid_lst=qid_lst, docids_lst=docids_lst, **kwargs
        )

    def retrieve(self, qid_lst, docids_lst, **kwargs):
        return self.retrieve_method.retrieve(
            buffer=self, qid_lst=qid_lst, docids_lst=docids_lst, **kwargs
        )

    def replace(self, **kwargs):
        return self.update_method.replace(buffer=self, **kwargs)

    def save(self, output_dir: str):
        output = open(os.path.join(output_dir, "buffer.pkl"), "wb")
        if self.params.update_method == "gss":
            pickle.dump(
                (
                    self.n_seen_so_far,
                    self.buffer_qid2dids,
                    self.update_method.buffer_score,
                ),
                output,
            )
        else:
            pickle.dump((self.n_seen_so_far, self.buffer_qid2dids), output)
        output.close()

    def update_old_embs(self, doc_ids, doc_embs):
        print(f"doc_ids:{len(doc_ids)}, doc_embs:{len(doc_embs)}, {doc_embs[0].shape}")
        for doc_id, doc_emb in zip(doc_ids, doc_embs):
            print(f"Save {doc_id} {doc_emb.shape}")
            self.buffer_did2emb[str(doc_id)] = doc_emb
