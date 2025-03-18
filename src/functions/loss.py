import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .similarities import calculate_S_qd_regl_logits


class InfoNCETermLoss(nn.Module):
    def __init__(self):
        super(InfoNCETermLoss, self).__init__()

    # (batch_size, 254, 768)  # (batch_size, positive_k + negative_k, 254, 768)
    def forward(self, q_embeddings, d_embeddings):
        # print(f"q_embeddings:{q_embeddings.shape}, d_embeddings:{d_embeddings.shape}")
        logits = calculate_S_qd_regl_logits(q_embeddings, d_embeddings)
        labels = torch.zeros(
            d_embeddings.size(0), d_embeddings.size(1), device=q_embeddings.device
        )
        labels[:, :1] = 1.0
        labels = labels.long()
        # print(f"logits:{logits.shape}/{logits}, labels:{labels.shape}")
        loss = F.cross_entropy(logits, labels)
        return loss


class InfoNCELoss(nn.Module):
    def __init__(self):
        super(InfoNCELoss, self).__init__()

    def forward(self, query_emb, positive_emb, negative_emb):
        if query_emb.dim() == 3:
            query_emb = query_emb.squeeze(1)
        # query_emb: (batch_size, embedding_dim)
        # positive_emb: (batch_size, positive_k, embedding_dim)
        # negative_emb: (batch_size, negative_k, embedding_dim)
        pos_sim = torch.matmul(
            query_emb.unsqueeze(1), positive_emb.transpose(-1, -2)
        ).squeeze(
            1
        )  # (batch_size, positive_k)
        neg_sim = torch.matmul(
            query_emb.unsqueeze(1), negative_emb.transpose(-1, -2)
        ).squeeze(
            1
        )  # (batch_size, negative_k)
        logits = torch.cat(
            (pos_sim, neg_sim), dim=1
        )  # (batch_size, positive_k + negative_k)
        labels = torch.zeros(
            query_emb.size(0), dtype=torch.long, device=query_emb.device
        )
        loss = F.cross_entropy(logits, labels)
        return loss


class SimpleContrastiveLoss:
    def __call__(
        self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = "mean"
    ):
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(
                0,
                x.size(0) * target_per_qry,
                target_per_qry,
                device=x.device,
                dtype=torch.long,
            )
        logits = torch.matmul(x, y.transpose(0, 1))
        return F.cross_entropy(logits, target, reduction=reduction)
