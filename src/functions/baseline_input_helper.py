from typing import Dict, List, Tuple, Optional, Any, Union
from transformers import BertTokenizer
import torch

max_q_len: int = 32
max_p_len: int = 128

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# https://github.com/caiyinqiong/L-2R/blob/main/src/tevatron/trainer.py


def _prepare_inputs(
    self, inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
) -> List[Dict[str, Union[torch.Tensor, Any]]]:
    prepared = []
    for x in inputs[2:]:
        for key, val in x.items():
            x[key] = val.to(self.args.device)
        prepared.append(x)

    if self.data_args.cl_method == "er":
        if not self.data_args.compatible:
            qid_lst, docids_lst = inputs[0], inputs[1]

            mem_passage = self.buffer.retrieve(
                qid_lst=qid_lst,
                docids_lst=docids_lst,
                q_lst=prepared[0],
                d_lst=prepared[1],
                lr=self._get_learning_rate(),
            )  # ER: [num_q * mem_bz, d_len], cpu; MIR: [num_q, mem_bz, d_len], gpu
            self.buffer.update(
                qid_lst=qid_lst,
                docids_lst=docids_lst,
                q_lst=prepared[0],
                d_lst=prepared[1],
            )

            if mem_passage is not None:
                for key, val in mem_passage.items():
                    passage_len = val.size(-1)
                    prepared[1][key] = prepared[1][key].reshape(
                        len(qid_lst), -1, passage_len
                    )  # [num_q, bz, d_len]
                    val = val.reshape(len(qid_lst), -1, passage_len).to(
                        prepared[1][key].device
                    )  # [num_q, mem_bz, d_len]
                    prepared[1][key] = torch.cat(
                        (prepared[1][key], val), dim=1
                    ).reshape(
                        -1, passage_len
                    )  # [num_q*(bz+mem_bz), d_len]
        else:
            qid_lst, docids_lst = inputs[0], inputs[1]

            mem_docids_lst, mem_passage = self.buffer.retrieve(
                qid_lst=qid_lst,
                docids_lst=docids_lst,
                q_lst=prepared[0],
                d_lst=prepared[1],
                lr=self._get_learning_rate(),
            )  # ER:[num_q * mem_bz],cpu, [num_q * mem_bz, d_len], cpu; MIR: MIR: [num_q, mem_bz],cpu, [num_q, mem_bz, d_len], gpu
            self.buffer.update(
                qid_lst=qid_lst,
                docids_lst=docids_lst,
                q_lst=prepared[0],
                d_lst=prepared[1],
            )

            if mem_passage is not None:
                for key, val in mem_passage.items():
                    passage_len = val.size(-1)
                    prepared[1][key] = prepared[1][key].reshape(
                        len(qid_lst), -1, passage_len
                    )  # [num_q, bz, d_len]
                    val = val.reshape(len(qid_lst), -1, passage_len).to(
                        prepared[1][key].device
                    )  # [num_q, mem_bz, d_len]
                    prepared[1][key] = torch.cat(
                        (prepared[1][key], val), dim=1
                    ).reshape(
                        -1, passage_len
                    )  # [num_q*(bz+mem_bz), d_len]

                docids_lst = torch.tensor(docids_lst).reshape(
                    len(qid_lst), -1
                )  # [num_q, n]
                mem_docids_lst = mem_docids_lst.reshape(
                    len(qid_lst), -1
                )  # [num_q, mem_bz]
                all_docids_lst = torch.cat(
                    (docids_lst, mem_docids_lst), dim=-1
                ).reshape(
                    -1
                )  # [num_q * n+mem_bz]

                identity = []
                doc_oldemb = []
                for i, docids in enumerate(all_docids_lst):
                    docids = int(docids)
                    if docids in self.buffer.buffer_did2emb:
                        identity.append(i)
                        doc_oldemb.append(self.buffer.buffer_did2emb[docids])
                identity = torch.tensor(identity)
                doc_oldemb = torch.tensor(np.array(doc_oldemb), device=self.args.device)
                prepared.append(identity)
                prepared.append(doc_oldemb)
    elif self.data_args.cl_method == "our":
        if not self.data_args.compatible:
            qid_lst, docids_lst = inputs[0], inputs[1]

            mem_passage, pos_docids, candidate_neg_docids = self.buffer.retrieve(
                qid_lst=qid_lst,
                docids_lst=docids_lst,
                q_lst=prepared[0],
                d_lst=prepared[1],
            )  # [num_q*(new_bz+mem_bz), d_len]
            self.buffer.update(
                qid_lst=qid_lst,
                docids_lst=docids_lst,
                pos_docids=pos_docids,
                candidate_neg_docids=candidate_neg_docids,
            )

            if mem_passage is not None:
                for key, val in mem_passage.items():
                    prepared[1][key] = val
        else:
            qid_lst, docids_lst = inputs[0], inputs[1]

            mem_emb, mem_passage, pos_docids, candidate_neg_docids = (
                self.buffer.retrieve(
                    qid_lst=qid_lst,
                    docids_lst=docids_lst,
                    q_lst=prepared[0],
                    d_lst=prepared[1],
                )
            )  # [num_q*(1+mem_bz), 768],gpu; [num_q*(new_bz+mem_bz), d_len],gpu
            self.buffer.update(
                qid_lst=qid_lst,
                docids_lst=docids_lst,
                pos_docids=pos_docids,
                candidate_neg_docids=candidate_neg_docids,
            )

            if mem_passage is not None:
                for key, val in mem_passage.items():
                    prepared[1][key] = val

                identity = []  # [1+mem_batch_size, num_q]
                pos_identity = torch.arange(len(qid_lst)) * (
                    1 + self.data_args.new_batch_size + self.data_args.mem_batch_size
                )
                identity.append(pos_identity)
                for i in range(self.data_args.mem_batch_size):
                    identity.append(
                        pos_identity + i + 1 + self.data_args.new_batch_size
                    )
                identity = torch.stack(identity, dim=0).transpose(0, 1).reshape(-1)
                prepared.append(identity)

                prepared.append(mem_emb)
    elif self.data_args.cl_method == "incre":
        if self.data_args.compatible:
            qid_lst, docids_lst = inputs[0], inputs[1]
            docids_lst = torch.tensor(docids_lst).reshape(
                len(qid_lst), -1
            )  # [num_q, n]

            identity = torch.arange(docids_lst.size(0)) * docids_lst.size(1)
            prepared.append(identity)

            doc_oldemb = []  # [num_q, 768]
            for docid in docids_lst[:, 0]:  # 对于incre，只有正例是old doc
                doc_oldemb.append(self.buffer.buffer_did2emb[int(docid)])
            doc_oldemb = torch.tensor(np.array(doc_oldemb)).to(self.args.device)
            prepared.append(doc_oldemb)
    else:
        print("not implement...")

    return prepared


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


def load_inputs(query_path, doc_path):
    pass


def prepare_inputs(query_path, doc_path):
    inputs = load_inputs(query_path, doc_path)
    prepared_inputs = _prepare_inputs(inputs)
    return prepared_inputs
