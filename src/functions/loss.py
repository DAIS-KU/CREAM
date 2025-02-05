import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, query_emb, positive_emb, negative_emb):
        # query_emb: (batch_size, embedding_dim)
        # positive_emb: (batch_size, positive_k, embedding_dim)
        # negative_emb: (batch_size, negative_k, embedding_dim)

        # 유사도 계산
        pos_sim = nn.functional.cosine_similarity(
            query_emb.unsqueeze(1), positive_emb, dim=-1
        )  # (batch_size, positive_k)
        neg_sim = nn.functional.cosine_similarity(
            query_emb.unsqueeze(1), negative_emb, dim=-1
        )  # (batch_size, negative_k)

        # 모든 유사도를 결합
        logits = (
            torch.cat((pos_sim, neg_sim), dim=1) / self.temperature
        )  # (batch_size, positive_k + negative_k)

        # 정답 레이블 생성: 각 쿼리의 첫 번째 positive 샘플을 정답으로 설정
        labels = torch.zeros(
            query_emb.size(0), dtype=torch.long, device=query_emb.device
        )

        # Cross-entropy loss 계산
        loss = F.cross_entropy(logits, labels)
        return loss
