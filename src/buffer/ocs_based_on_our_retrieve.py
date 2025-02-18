import torch
import torch.nn.functional as F
from .buffer_utils import random_retrieve, get_grad_vector, cosine_similarity
import copy
import numpy as np
import collections


class Our_retrieve(object):
    def __init__(self, params, train_params, **kwargs):
        super().__init__()
        self.params = params
        self.train_params = train_params
        
        self.alpha = params.alpha             # ocs 논문 상 1
        self.beta = params.beta               # ocs 논문 상 1
        self.gamma = params.gamma              # corset affinity 하이퍼 파라미터 (추가 필요)
        self.new_bz = params.new_batch_size     # 수정 필요. ocs는 new data에서만 selection
        # 하지만 training시 버퍼에서 random sampling한 B_c도 이용하므로 mem_upsample_docids을 model training에 이용할지 논의 필요

        self.mem_upsample = params.mem_upsample
        self.mem_bz = params.mem_batch_size       # 수정 필요

    def retrieve(self, buffer, qid_lst, docids_lst, **kwargs):
        model_temp = copy.deepcopy(buffer.model)      # f_t  <- f_(t-1) model
        grad_dims = []
        for param in model_temp.parameters():
            grad_dims.append(param.data.numel()) 
        
        batch_size = len(qid_lst) # 배치 쿼리 수
        n_doc = len(docids_lst) // len(qid_lst)  # 쿼리당 문서 수
        docids_pos_lst = np.array(docids_lst).reshape(batch_size, n_doc)[:,0]  # 첫번째 문서는 pos doc
        docids_neg_lst = np.array(docids_lst).reshape(batch_size, n_doc)[:,1:] 
        q_lst, d_lst = kwargs['q_lst'], kwargs['d_lst']   

        res_d_lst = {'input_ids':[], 'attention_mask': []}  
        res_pos_did_lst = collections.defaultdict(set)
        res_neg_did_lst = collections.defaultdict(set)
        for i, qid in enumerate(qid_lst):
            ############## 处理new data #############
            cur_q_lst = {}  # [1, 32], cuda
            for key, val in q_lst.items():
                cur_q_lst[key] = val[i:i+1]
            cur_d_lst = {}  # [n, 128], cuda
            for key, val in d_lst.items():
                cur_d_lst[key] = val.reshape(batch_size, n_doc, -1)[i]
            new_avg_grad, new_each_grad = self.get_batch_sim_new(model_temp, grad_dims, cur_q_lst, cur_d_lst)  # avg_grad 是个向量，each_grad=[new_upsample-1, 参数个数]
            # 한 특정 쿼리(qid)에 대해서 minibatch sim, samp div에 이용할 new_each_grad, new_avg_grad 계산
            
            ############### 处理mem data ##############
            mem_upsample = min(self.mem_upsample, len(buffer.buffer_qid2dids[qid])) 
            mem_bz = min(mem_upsample, self.mem_bz)
            if mem_bz == 0 or mem_upsample == 0:
                continue
            mem_upsample_docids = random_retrieve(buffer.buffer_qid2dids[qid], mem_upsample) # qid에 해당하는 메모리 버퍼에서 random sampling
            # model training을 위한 selection 시 여기서 또 mem_bz 수 만큼 random sampling??? 의논 필요
            mem_doc_lst = [buffer.did2doc[did] for did in mem_upsample_docids]
            mem_doc_lst = buffer.tokenizer.batch_encode_plus(mem_doc_lst,
                        add_special_tokens=True,
                        padding='max_length',
                        max_length=self.params.p_max_len,
                        truncation='only_first',
                        return_attention_mask=True,
                        return_token_type_ids=False,
                        return_tensors="pt")
            for key, value in mem_doc_lst.items():
                mem_doc_lst[key] = value.to(self.train_params.device)
            mem_avg_grad, mem_each_grad = self.get_batch_sim_mem(model_temp, grad_dims, cur_q_lst, cur_d_lst, mem_doc_lst)  # [new_upsample, 参数个数]
            index_new = self.get_new_data(new_avg_grad, new_each_grad, mem_avg_grad, self.new_bz, self.alpha, self.beta) # new_data에서 selection
            
            res_pos_did_lst[qid].add(docids_pos_lst[i])  
            res_neg_did_lst[qid].update(docids_neg_lst[i][index_new.cpu()]) 
            for key, val in cur_d_lst.items():
                res_d_lst[key].append(val[:1])  
                res_d_lst[key].append(val[index_new+1]) 
                
        #     for key, val in mem_doc_lst.items():
        #         res_d_lst[key].append(val[index_mem])  # 选择出来的memory负例
        # for key, val in res_d_lst.items():
        #     res_d_lst[key] = torch.cat(val, dim=0)
        return res_d_lst, res_pos_did_lst, res_neg_did_lst

    def get_batch_sim_mem(self, model_temp, grad_dims, q_lst, d_lst, mem_doc_lst):
        model_temp.zero_grad()            # ocs 논문처럼 메모리 버퍼 데이터도 현 모델 이용해 grad 구함
        loss = model_temp.forward(q_lst, mem_doc_lst, False).loss  # 현 쿼리(qid)와, qid에 대한 버퍼 데이터 input으로 loss 계산
        loss.backward()
        avg_grad = get_grad_vector(model_temp.parameters, grad_dims, self.train_params.device) 
        
        num_doc = mem_doc_lst['input_ids'].size(0)
        mem_grads = torch.zeros(num_doc, sum(grad_dims), dtype=torch.float32).to(self.train_params.device)
        for i in range(num_doc):
            doc_lst = {}
            for key, value in mem_doc_lst.items():
                doc_lst[key] = torch.cat((d_lst[key][:1], value[i:i+1]), dim=0)  # [2, 128] 
                       
            model_temp.zero_grad()
            loss = model_temp.forward(q_lst, doc_lst, False).loss  
            loss.backward()
            mem_grads[i].data.copy_(get_grad_vector(model_temp.parameters, grad_dims, self.train_params.device))
        return avg_grad, mem_grads

    # def get_mem_data(self, mem_each_grad, new_each_grad, mem_bz):
    #     inter_sim = cosine_similarity(mem_each_grad, new_each_grad)  # [mem_upsample, new_upsample-1]
    #     inter_sim_diag = torch.diag(inter_sim)
    #     inter_sim_sum = torch.sum(inter_sim, dim=-1)
    #     inter_sim = (inter_sim_sum - inter_sim_diag) * (-1.0) / (inter_sim.size(1)-1)  # [mem_upsample, 1]

    #     indexs = inter_sim.sort(dim=0, descending=True)[1][:mem_bz]
    #     return indexs
    
    def get_batch_sim_new(self, model_temp, grad_dims, q_lst, d_lst):
        model_temp.zero_grad()
        loss = model_temp.forward(q_lst, d_lst, False).loss  # f_t-1 모델에 현 쿼리(qid)에 대한 q,d 넣어서 loss 계산 후 grad 얻는 과정
        loss.backward()
        avg_grad = get_grad_vector(model_temp.parameters, grad_dims, self.train_params.device)
        num_doc = d_lst['input_ids'].size(0)
        mem_grads = torch.zeros(num_doc-1, sum(grad_dims), dtype=torch.float32).to(self.train_params.device)  # [num_mem_subs, grad_total_dims]

        for i in range(1, num_doc): # 정답 문서 제외
            doc_lst = {}
            for key, value in d_lst.items():
                doc_lst[key] = torch.cat((value[:1], value[i:i+1]), dim=0)  # [2, 128] # pos doc 합쳐서 loss 계산

            model_temp.zero_grad()
            loss = model_temp.forward(q_lst, doc_lst, False).loss  # 여기선 각 문서에 대해 진행 
            loss.backward()
            mem_grads[i-1].data.copy_(get_grad_vector(model_temp.parameters, grad_dims, self.train_params.device))
            #현재 문서(i)의 gradient를 벡터로 변환하여 mem_grads에 저장 (pos 제외)
        return avg_grad, mem_grads

    def get_new_data(self,new_avg_grad, new_each_grad, mem_avg_grad, new_bz, alpha, beta, gamma):
        minibatch_sim = cosine_similarity(new_each_grad, new_avg_grad.unsqueeze(0)).squeeze(dim=-1)  # [new_upsample-1]

        samp_div = cosine_similarity(new_each_grad, new_each_grad)  # [new_upsample-1, new_upsample-1]
        samp_div_diag = torch.diag(inter_sim)
        samp_div_sum = torch.sum(inter_sim, dim=-1)
        samp_div = (samp_div_sum - samp_div_diag) * (-1.0) / (samp_div.size(1)-1)  # [new_upsample-1] 
        # [-1,0] 사이 값을 가지고 0에 가까울수록 div 높음

        coreset_sim = cosine_similarity(new_each_grad, mem_avg_grad.unsqueeze(0)).squeeze(dim=-1) # new_coming 데이터가 기존 버퍼 데이터와 유사한 정도
        
        sim = alpha * minibatch_sim + beta * samp_div + gamma * coreset_sim
        indexs = sim.sort(dim=0, descending=True)[1][:new_bz]
        return indexs
