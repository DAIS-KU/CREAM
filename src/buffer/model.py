# l2r포함 l2r baseline들은 기본 bert 외 정의한 dense model 사용
# https://github.com/caiyinqiong/L-2R/blob/022f1282cd5bdf42c29d0f006e5c1b77c2c5c724/src/tevatron/modeling/encoder.py#L81
# https://github.com/caiyinqiong/L-2R/blob/022f1282cd5bdf42c29d0f006e5c1b77c2c5c724/src/tevatron/modeling/dense.py#L29

import copy
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel, PreTrainedModel
from transformers.file_utils import ModelOutput

from transformers import AutoTokenizer
import string

from .arguments import ModelArguments
from .arguments import TevatronTrainingArguments as TrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class EncoderPooler(nn.Module):
    def __init__(self, **kwargs):
        super(EncoderPooler, self).__init__()
        self._config = {}

    def forward(self, q_reps, p_reps):
        raise NotImplementedError("EncoderPooler is an abstract class")

    def load(self, model_dir: str):
        pooler_path = os.path.join(model_dir, "pooler.pt")
        if pooler_path is not None:
            if os.path.exists(pooler_path):
                logger.info(f"Loading Pooler from {pooler_path}")
                state_dict = torch.load(pooler_path, map_location="cpu")
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, "pooler.pt"))
        with open(os.path.join(save_path, "pooler_config.json"), "w") as f:
            json.dump(self._config, f)


class EncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(
        self,
        lm_q: PreTrainedModel,
        lm_p: PreTrainedModel,
        pooler: nn.Module = None,
        untie_encoder: bool = False,
        negatives_x_device: bool = False,
        compatible_ce_alpha: float = 0.0,
    ):
        super().__init__()
        self.lm_q = lm_q.to("cuda:0")
        self.lm_p = lm_p.to("cuda:0")
        self.pooler = pooler
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.compatible = nn.CrossEntropyLoss(reduction="mean")  # compatible loss

        self.compatible_ce_alpha = compatible_ce_alpha
        self.negatives_x_device = negatives_x_device
        self.untie_encoder = untie_encoder
        if self.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError(
                    "Distributed training has not been initialized for representation all gather."
                )
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    # model_temp.forward(q_lst, d_lst, False)???
    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
        identity: Tensor = None,
        oldemb: Tensor = None,
    ):
        # print("identity: ", identity)
        # print("oldemb: ", oldemb)
        q_reps = self.encode_query(query)
        p_reps = self.encode_passage(passage)

        # for inference
        if q_reps is None or p_reps is None:
            if identity is not None:
                p_reps[identity] = oldemb.clone()
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

        # for training
        if self.training:
            if self.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
            if identity is not None:
                target_fullupdate = self.compute_similarity(q_reps, p_reps).view(
                    q_reps.size(0), -1
                )
                p_reps[identity] = oldemb.clone()

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(
                scores.size(0), device=scores.device, dtype=torch.long
            )
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores, target)
            if identity is not None:
                compatible_loss = self.compute_compatiblelogit_loss(
                    scores, target_fullupdate
                )
                # print(loss, compatible_loss)
                loss = loss + self.compatible_ce_alpha * compatible_loss
            if self.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
        # for eval
        else:
            if identity is not None:
                p_reps[identity] = oldemb.clone()
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    @staticmethod
    def build_pooler(model_args):
        return None

    @staticmethod
    def load_pooler(weights, **config):
        return None

    def encode_passage(self, psg):
        raise NotImplementedError("EncoderModel is an abstract class")

    def encode_query(self, qry):
        raise NotImplementedError("EncoderModel is an abstract class")

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def compute_compatible_loss(self, newemb, oldemb):
        return torch.mean(self.distance(newemb, oldemb))

    def compute_compatiblelogit_loss(self, scores, target):
        target = target.softmax(dim=-1)
        return self.compatible(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        train_args: TrainingArguments,
        **hf_kwargs,
    ):
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:
                _qry_model_path = os.path.join(
                    model_args.model_name_or_path, "query_model"
                )
                _psg_model_path = os.path.join(
                    model_args.model_name_or_path, "passage_model"
                )
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f"loading query model weight from {_qry_model_path}")
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(_qry_model_path, **hf_kwargs)
                logger.info(f"loading passage model weight from {_psg_model_path}")
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(_psg_model_path, **hf_kwargs)
            else:
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    model_args.model_name_or_path, **hf_kwargs
                )
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name_or_path, **hf_kwargs
            )
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            negatives_x_device=train_args.negatives_x_device,
            untie_encoder=model_args.untie_encoder,
            compatible_ce_alpha=train_args.compatible_ce_alpha,
        )
        return model

    @classmethod
    def load(
        cls,
        model_name_or_path,
        **hf_kwargs,
    ):
        # load local
        untie_encoder = True
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, "query_model")
            _psg_model_path = os.path.join(model_name_or_path, "passage_model")
            if os.path.exists(_qry_model_path):
                logger.info(f"found separate weight for query/passage encoders")
                logger.info(f"loading query model weight from {_qry_model_path}")
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(_qry_model_path, **hf_kwargs)
                logger.info(f"loading passage model weight from {_psg_model_path}")
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(_psg_model_path, **hf_kwargs)
                untie_encoder = False
            else:
                logger.info(f"try loading tied weight")
                logger.info(f"loading model weight from {model_name_or_path}")
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    model_name_or_path, **hf_kwargs
                )
                lm_p = lm_q
        else:
            logger.info(f"try loading tied weight")
            logger.info(f"loading model weight from {model_name_or_path}")
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
            lm_p = lm_q

        pooler_weights = os.path.join(model_name_or_path, "pooler.pt")
        pooler_config = os.path.join(model_name_or_path, "pooler_config.json")
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f"found pooler weight and configuration")
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.load_pooler(model_name_or_path, **pooler_config_dict)
        else:
            pooler = None

        model = cls(lm_q=lm_q, lm_p=lm_p, pooler=pooler, untie_encoder=untie_encoder)
        return model

    def save(self, output_dir: str):
        if self.untie_encoder:
            os.makedirs(os.path.join(output_dir, "query_model"))
            os.makedirs(os.path.join(output_dir, "passage_model"))
            self.lm_q.save_pretrained(os.path.join(output_dir, "query_model"))
            self.lm_p.save_pretrained(os.path.join(output_dir, "passage_model"))
        else:
            self.lm_q.save_pretrained(output_dir)
        if self.pooler:
            self.pooler.save_pooler(output_dir)


class DenseModel(EncoderModel):
    def mean_pooling(self, hidden_states, attention_mask):
        """패딩 제외하고 mean pooling 적용"""
        mask_expanded = attention_mask.unsqueeze(-1).expand(
            hidden_states.shape
        )  # (batch_size, seq_len, hidden_dim)
        sum_hidden = (hidden_states * mask_expanded).sum(
            dim=1
        )  # 마스크된 부분 제외하고 합산
        sum_mask = mask_expanded.sum(
            dim=1
        )  # 마스크된 개수 계산 (길이별로 다를 수 있음)
        pooled_output = sum_hidden / sum_mask  # 평균값 반환
        return F.normalize(pooled_output, p=2, dim=1)  # L2 정규화 적용

    def encode_mean_pooling(self, input):
        if input is None:
            return None
        psg_out = self.lm_p(**input, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        p_reps = self.mean_pooling(p_hidden, input["attention_mask"])
        return p_reps

    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        if self.pooler is not None:
            p_reps = self.pooler(p=p_hidden)  # D * d
        else:
            p_reps = p_hidden[:, 0]
        return p_reps

    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out.last_hidden_state
        if self.pooler is not None:
            q_reps = self.pooler(q=q_hidden)
        else:
            q_reps = q_hidden[:, 0]  # CLS
        return q_reps

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))


class ColbertPooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, output_dim: int = 32, tied=True):
        super(ColbertPooler, self).__init__()
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)
        self._config = {"input_dim": input_dim, "output_dim": output_dim, "tied": tied}

    def forward(self, q: Tensor = None, p: Tensor = None, **kwargs):
        if q is not None:
            return self.linear_q(q)
        elif p is not None:
            return self.linear_p(p)
        else:
            raise ValueError


class ColbertModel(EncoderModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f"self.training: {self.training}")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[Q]", "[D]"]})
        self.skiplist = self.tokenizer.encode(
            string.punctuation, add_special_tokens=False
        )
        if self.lm_q:
            self.lm_q.resize_token_embeddings(len(self.tokenizer))
        if self.lm_p:
            self.lm_p.resize_token_embeddings(len(self.tokenizer))

    def mask(self, input_ids):
        mask = [
            [(x not in self.skiplist) and (x != 0) for x in d]
            for d in input_ids.cpu().tolist()
        ]
        return mask

    def encode_passage(self, psg):
        if psg is None:
            return None, None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        p_reps = self.pooler(p=p_hidden)
        mask = torch.tensor(self.mask(psg["input_ids"]), device=psg["input_ids"].device)
        p_reps *= mask[:, :, None].float()
        return p_reps, mask

    def encode_query(self, qry):
        if qry is None:
            return None, None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out.last_hidden_state
        q_reps = self.pooler(q=q_hidden)
        q_reps *= qry["attention_mask"][:, :, None].float()
        return q_reps, qry["attention_mask"]

    def compute_similarity(self, q_reps, p_reps):
        if self.training:
            token_scores = torch.einsum("qin,pjn->qipj", q_reps, p_reps)
            scores, _ = token_scores.max(-1)
            scores = scores.sum(1)
        else:
            scores = (q_reps @ p_reps.permute(0, 2, 1)).max(dim=2).values.sum(dim=1)
        return scores

    def forward(
        self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None
    ):
        q_reps, q_mask = self.encode_query(query)
        p_reps, p_mask = self.encode_passage(passage)

        # for inference
        if q_reps is None or p_reps is None:
            if q_reps is not None:  # query
                q_mask = q_mask.bool()
                q_reps = [emb[q_mask[idx]] for idx, emb in enumerate(q_reps)]
                q_len = [m.sum() for m in q_mask]
                return EncoderOutput(q_reps=q_reps, p_reps=p_reps, scores=q_len)
            if p_reps is not None:  # passage
                p_mask = p_mask.bool()
                p_reps = [emb[p_mask[idx]] for idx, emb in enumerate(p_reps)]
                p_len = [m.sum() for m in p_mask]
                return EncoderOutput(q_reps=q_reps, p_reps=p_reps, scores=p_len)

        # for training
        if self.training:
            if self.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(
                scores.size(0), device=scores.device, dtype=torch.long
            )
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores, target)
            if self.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction

        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    @staticmethod
    def load_pooler(model_weights_file, **config):
        pooler = ColbertPooler(**config)
        pooler.load(model_weights_file)
        return pooler

    @staticmethod
    def build_pooler(model_args):
        pooler = ColbertPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder,
        )
        pooler.load(model_args.model_name_or_path)
        return pooler
