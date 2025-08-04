import collections
import copy

import numpy as np
import torch
import torch.nn.functional as F

from .buffer_utils import cosine_similarity, get_grad_vector


# gss retrieval random
# https://github.com/caiyinqiong/L-2R/blob/main/src/tevatron/arguments.py#L48
class GSSGreedyUpdate(object):
    def __init__(self, params, train_params, **kwargs):
        super().__init__()
        self.mem_strength = params.gss_mem_strength  # 从memory中采样几个batch
        self.gss_batch_size = params.gss_batch_size  # 从memory中采样时每个batch的大小
        if kwargs["buffer_score"]:
            print("load buffer score...")
            self.buffer_score = kwargs["buffer_score"]  # 存储每个样本附带的score
        else:
            print("init buffer score...")
            self.buffer_score = collections.defaultdict(list)  # 存储每个样本附带的score
        self.params = params
        self.train_params = train_params

    def update(self, buffer, qid_lst, docids_lst, **kwargs):
        print("Called GSSGreedyUpdate.update()")
        batch_size = len(qid_lst)
        n_doc = len(docids_lst) // len(qid_lst)
        docids_neg_lst = np.array(docids_lst).reshape(batch_size, n_doc)[
            :, 1:
        ]  # 去掉pos passage
        docids_pos_lst = np.array(docids_lst).reshape(batch_size, n_doc)[
            :, 0
        ]  # pos passage id

        model_temp = copy.deepcopy(buffer.model).to(self.train_params.device)

        grad_dims = []
        for param in model_temp.parameters():
            grad_dims.append(param.data.numel())

        q_lst, d_lst = kwargs["q_lst"], kwargs["d_lst"]

        # print("model_temp.lm_p.device:", next(model_temp.lm_p.parameters()).device)
        # print("buffer.model.lm_p.device:", next(buffer.model.lm_p.parameters()).device)
        # print("train_params.device:", self.train_params.device)
        # print("q_lst devices:", [v.device for v in q_lst.values()])
        # print("d_lst devices:", [v.device for v in d_lst.values()])

        for i, qid in enumerate(qid_lst):
            cur_q_lst = {}  # [1, 32], cuda
            for key, val in q_lst.items():
                cur_q_lst[key] = val[i : i + 1]
            cur_d_lst = {}  # [n, 128], cuda
            for key, val in d_lst.items():
                cur_d_lst[key] = val.reshape(batch_size, n_doc, -1)[i]
            docids = docids_neg_lst[i]  # 该query新来的 negative doc, array
            place_left = max(0, buffer.buffer_size - len(buffer.buffer_qid2dids[qid]))
            if place_left <= 0:  # 如果此时buffer已满
                batch_sim, mem_grads = self.get_batch_sim(
                    buffer,
                    model_temp,
                    grad_dims,
                    qid,
                    docids_pos_lst[i],
                    cur_q_lst,
                    cur_d_lst,
                )  # 从buffer随机采样几个batch计算他们的梯度，并计算batch_x整体与他们的梯度最大相似度， batch_sim=实数，mem_grads=[mem_strength,模型总参数量]
                """
                x는 buffer에서 임의로 몇 개의 배치를 샘플링하여 그들의 **기울기(gradient)**를 계산하고, batch_x와 그 샘플링된 배치들의 기울기 간 최대 유사도를 계산합니다. 
                이때 batch_sim은 계산된 유사도 값(실수)이고, mem_grads는 **기억 강도(mem_strength)**와 모델의 총 파라미터 수를 포함하는 리스트입니다.
                """
                if batch_sim < 0:
                    if len(self.buffer_score[qid]) == 0:
                        buffer_score = torch.ones(len(docids)).to(
                            self.train_params.device
                        )
                        batch_item_sim = self.get_each_batch_sample_sim(
                            model_temp, grad_dims, mem_grads, cur_q_lst, cur_d_lst
                        )
                        buffer.buffer_qid2dids[qid] = [str(did) for did in docids]
                        self.buffer_score[qid] = batch_item_sim.clone().cpu().tolist()
                        continue
                    else:
                        buffer_score = torch.Tensor(self.buffer_score[qid]).to(
                            self.train_params.device
                        )

                    # if len(self.buffer_score[qid]) == 0:
                    #     # 안전하게 초기화
                    #     buffer_score = torch.ones(len(docids)).to(self.train_params.device)
                    # else:
                    #     buffer_score = torch.Tensor(self.buffer_score[qid]).to(self.train_params.device)

                    buffer_sim = (buffer_score - torch.min(buffer_score)) / (
                        (torch.max(buffer_score) - torch.min(buffer_score)) + 0.01
                    )

                    if buffer_sim.sum() == 0 or torch.isnan(buffer_sim).any():
                        print(
                            f"buffer_sim is invalid (sum={buffer_sim.sum().item():.4f}), using uniform distribution as fallback."
                        )
                        buffer_sim = torch.ones_like(buffer_sim) / buffer_sim.size(0)

                    index = torch.multinomial(
                        buffer_sim, len(docids), replacement=False
                    )

                    batch_item_sim = self.get_each_batch_sample_sim(
                        model_temp, grad_dims, mem_grads, cur_q_lst, cur_d_lst
                    )  # 计算每个new data与mem_grads的梯度最大相似度， batch_sample_memory_cos=[len(x)]
                    scaled_batch_item_sim = ((batch_item_sim + 1) / 2).unsqueeze(
                        1
                    )  # 标准化到[0,1]
                    buffer_repl_batch_sim = ((buffer_score[index] + 1) / 2).unsqueeze(1)
                    outcome = torch.multinomial(
                        torch.cat(
                            (scaled_batch_item_sim, buffer_repl_batch_sim), dim=1
                        ),
                        1,
                        replacement=False,
                    )
                    # 수정 후
                    sub_index = outcome.squeeze(1).bool()  # 由outcome决定是否替换
                    added_indx = torch.arange(
                        end=batch_item_sim.size(0), device=sub_index.device
                    )

                    # 执行替换
                    buffer.buffer_qid2dids[qid] = np.array(buffer.buffer_qid2dids[qid])
                    self.buffer_score[qid] = np.array(self.buffer_score[qid])

                    #
                    # buffer.buffer_qid2dids[qid][index[sub_index].cpu()] = docids[
                    #     added_indx[sub_index].cpu()
                    # ].copy()
                    # self.buffer_score[qid][index[sub_index].cpu()] = (
                    #     batch_item_sim[added_indx[sub_index].cpu()]
                    #     .clone()
                    #     .cpu()
                    #     .numpy()
                    # )
                    #

                    index_cpu = index[sub_index].cpu()
                    added_indx_cpu = added_indx[sub_index].cpu()

                    valid_len = len(buffer.buffer_qid2dids[qid])
                    valid_mask = index_cpu < valid_len

                    index_cpu = index_cpu[valid_mask]
                    added_indx_cpu = added_indx_cpu[valid_mask]

                    if len(index_cpu) > 0:
                        buffer.buffer_qid2dids[qid][index_cpu] = docids[
                            added_indx_cpu
                        ].copy()
                        self.buffer_score[qid][index_cpu] = (
                            batch_item_sim[added_indx_cpu].clone().cpu().numpy()
                        )

                    buffer.buffer_qid2dids[qid] = buffer.buffer_qid2dids[qid].tolist()
                    self.buffer_score[qid] = self.buffer_score[qid].tolist()
            else:  # 如果此时buffer未满，优先拿x中靠前的数据填满buffer，剩余的丢弃
                offset = min(place_left, len(docids))
                docids = docids[:offset]  # array
                if len(buffer.buffer_qid2dids[qid]) == 0:
                    batch_sample_memory_cos = (
                        torch.zeros(len(docids)) + 0.1
                    )  # 初始化score
                else:
                    mem_grads = self.get_rand_mem_grads(
                        buffer, model_temp, grad_dims, qid, docids_pos_lst[i], cur_q_lst
                    )  # 从buffer随机采样几个batch计算他们的梯度， mem_grads=[mem_strength,模型总参数量]
                    batch_sample_memory_cos = self.get_each_batch_sample_sim(
                        model_temp, grad_dims, mem_grads, cur_q_lst, cur_d_lst
                    )  # 计算每个new data与mem_grads的梯度最大相似度， batch_sample_memory_cos=[len(x)]
                buffer.buffer_qid2dids[qid].extend(str(did) for did in docids)
                self.buffer_score[qid].extend(batch_sample_memory_cos.tolist())

    '''
    def update(self, buffer, qid_lst, docids_lst, **kwargs):
        print("Called GSSGreedyUpdate.update()")
        batch_size = len(qid_lst)
        n_doc = len(docids_lst) // len(qid_lst)
        docids_neg_lst = np.array(docids_lst).reshape(batch_size, n_doc)[
            :, 1:
        ]  # 去掉pos passage
        docids_pos_lst = np.array(docids_lst).reshape(batch_size, n_doc)[
            :, 0
        ]  # pos passage id

        model_temp = copy.deepcopy(buffer.model)
        grad_dims = []
        for param in model_temp.parameters():
            grad_dims.append(param.data.numel())

        q_lst, d_lst = kwargs["q_lst"], kwargs["d_lst"]
        for i, qid in enumerate(qid_lst):
            cur_q_lst = {}  # [1, 32], cuda
            for key, val in q_lst.items():
                cur_q_lst[key] = val[i : i + 1]
            cur_d_lst = {}  # [n, 128], cuda
            for key, val in d_lst.items():
                cur_d_lst[key] = val.reshape(batch_size, n_doc, -1)[i]
            docids = docids_neg_lst[i]  # 该query新来的 negative doc, array
            place_left = max(0, buffer.buffer_size - len(buffer.buffer_qid2dids[qid]))
            if place_left <= 0:  # 如果此时buffer已满
                batch_sim, mem_grads = self.get_batch_sim(
                    buffer,
                    model_temp,
                    grad_dims,
                    qid,
                    docids_pos_lst[i],
                    cur_q_lst,
                    cur_d_lst,
                )  # 从buffer随机采样几个batch计算他们的梯度，并计算batch_x整体与他们的梯度最大相似度， batch_sim=实数，mem_grads=[mem_strength,模型总参数量]
                """
                x는 buffer에서 임의로 몇 개의 배치를 샘플링하여 그들의 **기울기(gradient)**를 계산하고, batch_x와 그 샘플링된 배치들의 기울기 간 최대 유사도를 계산합니다. 
                이때 batch_sim은 계산된 유사도 값(실수)이고, mem_grads는 **기억 강도(mem_strength)**와 모델의 총 파라미터 수를 포함하는 리스트입니다.
                """
                if batch_sim < 0:
                    buffer_score = torch.Tensor(self.buffer_score[qid]).to(
                        self.train_params.device
                    )  # tensor
                    buffer_sim = (buffer_score - torch.min(buffer_score)) / (
                        (torch.max(buffer_score) - torch.min(buffer_score)) + 0.01
                    )
                    index = torch.multinomial(
                        buffer_sim, len(docids), replacement=False
                    )  # 按照标准化后的score采样出len(docids)个下标, tensor

                    batch_item_sim = self.get_each_batch_sample_sim(
                        model_temp, grad_dims, mem_grads, cur_q_lst, cur_d_lst
                    )  # 计算每个new data与mem_grads的梯度最大相似度， batch_sample_memory_cos=[len(x)]
                    scaled_batch_item_sim = ((batch_item_sim + 1) / 2).unsqueeze(
                        1
                    )  # 标准化到[0,1]
                    buffer_repl_batch_sim = ((buffer_score[index] + 1) / 2).unsqueeze(1)
                    outcome = torch.multinomial(
                        torch.cat(
                            (scaled_batch_item_sim, buffer_repl_batch_sim), dim=1
                        ),
                        1,
                        replacement=False,
                    )
                    added_indx = torch.arange(end=batch_item_sim.size(0))
                    sub_index = outcome.squeeze(1).bool()  # 由outcome决定是否替换

                    # 执行替换
                    buffer.buffer_qid2dids[qid] = np.array(buffer.buffer_qid2dids[qid])
                    self.buffer_score[qid] = np.array(self.buffer_score[qid])
                    buffer.buffer_qid2dids[qid][index[sub_index].cpu()] = docids[
                        added_indx[sub_index].cpu()
                    ].copy()
                    self.buffer_score[qid][index[sub_index].cpu()] = (
                        batch_item_sim[added_indx[sub_index].cpu()]
                        .clone()
                        .cpu()
                        .numpy()
                    )
                    buffer.buffer_qid2dids[qid] = buffer.buffer_qid2dids[qid].tolist()
                    self.buffer_score[qid] = self.buffer_score[qid].tolist()
            else:  # 如果此时buffer未满，优先拿x中靠前的数据填满buffer，剩余的丢弃
                offset = min(place_left, len(docids))
                docids = docids[:offset]  # array
                if len(buffer.buffer_qid2dids[qid]) == 0:
                    batch_sample_memory_cos = torch.zeros(len(docids)) + 0.1  # 初始化score
                else:
                    mem_grads = self.get_rand_mem_grads(
                        buffer, model_temp, grad_dims, qid, docids_pos_lst[i], cur_q_lst
                    )  # 从buffer随机采样几个batch计算他们的梯度， mem_grads=[mem_strength,模型总参数量]
                    batch_sample_memory_cos = self.get_each_batch_sample_sim(
                        model_temp, grad_dims, mem_grads, cur_q_lst, cur_d_lst
                    )  # 计算每个new data与mem_grads的梯度最大相似度， batch_sample_memory_cos=[len(x)]
                buffer.buffer_qid2dids[qid].extend(docids.tolist())
                self.buffer_score[qid].extend(batch_sample_memory_cos.tolist())
    '''

    def get_batch_sim(self, buffer, model_temp, grad_dims, qid, did_pos, q_lst, d_lst):
        mem_grads = self.get_rand_mem_grads(
            buffer, model_temp, grad_dims, qid, did_pos, q_lst
        )

        model_temp.zero_grad()
        # identity, old_emb는 compatible시 사용하는 것인데, gss는 compatible하지 않으므로
        # (q_lst, doc_lst, False)에서 (q_lst, doc_lst, None, None)으로 수정
        loss = model_temp.forward(q_lst, d_lst, None, None).loss  # 不使用inbatch_loss
        loss.backward()
        batch_grad = get_grad_vector(
            model_temp.parameters, grad_dims, self.train_params.device
        ).unsqueeze(0)
        batch_sim = max(cosine_similarity(mem_grads, batch_grad))
        return batch_sim, mem_grads

    """
    def get_rand_mem_grads(self, buffer, model_temp, grad_dims, qid, did_pos, q_lst):
        # 从memory中随机采mem_strength个batch，batch_size大小为gss_batch_size，计算他们的梯度

        buffer_docid_lst = buffer.buffer_qid2dids[qid]  # list
        gss_batch_size = min(self.gss_batch_size, len(buffer_docid_lst))  # batch size
        num_mem_subs = min(
            self.mem_strength, len(buffer_docid_lst) // gss_batch_size
        )  # 采几个batch
        mem_grads = torch.zeros(num_mem_subs, sum(grad_dims), dtype=torch.float32).to(
            self.train_params.device
        )  # [num_mem_subs, grad_total_dims]
        shuffeled_inds = torch.randperm(len(buffer_docid_lst))

        for i in range(num_mem_subs):
            random_batch_inds = shuffeled_inds[
                i * gss_batch_size : i * gss_batch_size + gss_batch_size
            ]
            docids_lst = np.array(buffer_docid_lst)[random_batch_inds]
            docids_lst = np.insert(docids_lst, 0, did_pos)

            doc_lst = [buffer.did2doc[str(did)] for did in docids_lst]

            for key, val in q_lst.items(): # 수정함
                q_lst[key] = val.to(self.train_params.device)
                
            doc_lst = buffer.tokenizer.batch_encode_plus(
                doc_lst,
                add_special_tokens=True,
                padding="max_length",
                max_length=self.params.p_max_len,
                truncation="only_first",
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            for key, value in doc_lst.items():
                doc_lst[key] = value.to(
                    self.train_params.device
                )  # [gss_batch_size+1, 128]

            model_temp.zero_grad()
            # identity, old_emb는 compatible시 사용하는 것인데, gss는 compatible하지 않으므로
            # (q_lst, doc_lst, False)에서 (q_lst, doc_lst, None, None)으로 수정
            loss = model_temp.forward(
                q_lst, doc_lst, None, None
            ).loss  # 不使用inbatch_loss
            loss.backward()
            mem_grads[i].data.copy_(
                get_grad_vector(
                    model_temp.parameters, grad_dims, self.train_params.device
                )
            )
        return mem_grads
        """

    # 수정한 부분
    def get_rand_mem_grads(self, buffer, model_temp, grad_dims, qid, did_pos, q_lst):
        # 从memory中随机采mem_strength个batch，batch_size大小为gss_batch_size，计算他们的梯度

        buffer_docid_lst = buffer.buffer_qid2dids[qid]  # list
        gss_batch_size = min(self.gss_batch_size, len(buffer_docid_lst))  # batch size
        num_mem_subs = min(
            self.mem_strength, len(buffer_docid_lst) // gss_batch_size
        )  # 采几个batch

        # 유효한 배치 수를 추적하기 위한 변수
        valid_batch_count = 0
        mem_grads = torch.zeros(num_mem_subs, sum(grad_dims), dtype=torch.float32).to(
            self.train_params.device
        )  # [num_mem_subs, grad_total_dims]

        shuffeled_inds = torch.randperm(len(buffer_docid_lst))

        for i in range(num_mem_subs):
            random_batch_inds = shuffeled_inds[
                i * gss_batch_size : i * gss_batch_size + gss_batch_size
            ]
            docids_lst = np.array(buffer_docid_lst)[random_batch_inds]
            docids_lst = np.insert(docids_lst, 0, did_pos)

            # 안전한 문서 조회로 수정
            doc_lst = []
            valid_docids = []

            for did in docids_lst:
                str_did = str(did)
                if str_did in buffer.did2doc:
                    doc_lst.append(buffer.did2doc[str_did])
                    valid_docids.append(str_did)
                else:
                    print(f"Warning: Document ID {str_did} not found in buffer.did2doc")
                    # 빈 문서로 대체하거나 스킵할 수 있음
                    # 여기서는 빈 문서로 대체
                    doc_lst.append("")

            # 유효한 문서가 없거나 모두 빈 문서인 경우 스킵
            if not doc_lst or all(doc == "" for doc in doc_lst):
                print(f"No valid documents for batch {i}, skipping...")
                continue

            # 최소 2개 문서(positive + negative)가 필요
            if len([doc for doc in doc_lst if doc != ""]) < 2:
                print(
                    f"Insufficient valid documents for batch {i} (need at least 2), skipping..."
                )
                continue

            # 쿼리 텐서를 디바이스로 이동
            for key, val in q_lst.items():
                q_lst[key] = val.to(self.train_params.device)

            try:
                # 문서 토크나이징
                doc_lst_tokenized = buffer.tokenizer.batch_encode_plus(
                    doc_lst,
                    add_special_tokens=True,
                    padding="max_length",
                    max_length=self.params.p_max_len,
                    truncation="only_first",
                    return_attention_mask=True,
                    return_token_type_ids=False,
                    return_tensors="pt",
                )

                for key, value in doc_lst_tokenized.items():
                    doc_lst_tokenized[key] = value.to(
                        self.train_params.device
                    )  # [gss_batch_size+1, 128]

                model_temp.zero_grad()
                # identity, old_emb는 compatible시 사용하는 것인데, gss는 compatible하지 않으므로
                # (q_lst, doc_lst, False)에서 (q_lst, doc_lst, None, None)으로 수정
                loss = model_temp.forward(
                    q_lst, doc_lst_tokenized, None, None
                ).loss  # 不使用inbatch_loss

                loss.backward()
                mem_grads[valid_batch_count].data.copy_(
                    get_grad_vector(
                        model_temp.parameters, grad_dims, self.train_params.device
                    )
                )
                valid_batch_count += 1

            except Exception as e:
                print(f"Error processing batch {i}: {str(e)}")
                continue

        # 유효한 배치가 없는 경우 기본값 반환
        if valid_batch_count == 0:
            print("No valid batches processed, returning zero gradients")
            return torch.zeros(1, sum(grad_dims), dtype=torch.float32).to(
                self.train_params.device
            )

        # 유효한 배치만 반환
        return mem_grads[:valid_batch_count]

    def get_each_batch_sample_sim(self, model_temp, grad_dims, mem_grads, q_lst, d_lst):
        # mem_grads是从memory中采样几个batch计算的几个梯度
        # 该函数用于计算batch_docids中每个新数据与mem_grads的cos最大值

        num_doc = d_lst["input_ids"].size(0)
        cosine_sim = torch.zeros(num_doc - 1).to(self.train_params.device)
        for i in range(1, num_doc):
            doc_lst = {}
            for key, value in d_lst.items():
                doc_lst[key] = torch.cat(
                    (value[:1], value[i : i + 1]), dim=0
                )  # [2, 128]

            model_temp.zero_grad()
            # identity, old_emb는 compatible시 사용하는 것인데, gss는 compatible하지 않으므로
            # (q_lst, doc_lst, False)에서 (q_lst, doc_lst, None, None)으로 수정
            loss = model_temp.forward(
                q_lst, doc_lst, None, None
            ).loss  # 不使用inbatch_loss
            loss.backward()
            this_grad = get_grad_vector(
                model_temp.parameters, grad_dims, self.train_params.device
            ).unsqueeze(0)
            cosine_sim[i - 1] = max(cosine_similarity(mem_grads, this_grad))
        return cosine_sim
