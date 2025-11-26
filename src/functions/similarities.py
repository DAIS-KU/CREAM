import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_S_qd_regl(E_q, E_d, device):
    # E_q(qlen, 768), E_d(dlen, 768),
    if isinstance(E_q, list):
        E_q = torch.stack(E_q, dim=0)
    if isinstance(E_d, list):
        E_d = torch.stack(E_d, dim=0)
    E_q = E_q.to(device).float()
    E_d = E_d.to(device).float()
    E_q_normalized = torch.nn.functional.normalize(E_q, p=2, dim=1)
    E_d_normalized = torch.nn.functional.normalize(E_d, p=2, dim=1)
    cosine_sim_matrix = torch.matmul(E_q_normalized, E_d_normalized.T)
    max_scores, _ = torch.max(cosine_sim_matrix, dim=1)
    S_qd_score = max_scores.sum()
    # print(f"E_q:{E_q.shape}, E_d:{E_d.shape}, max_scores:{max_scores.shape}, S_qd_score: {S_qd_score}")
    E_q = E_q.cpu()
    E_d = E_d.cpu()
    torch.cuda.empty_cache()
    return S_qd_score


def calculate_S_qd_regl_batch(E_q, E_d, device):
    # E_q(batch_size, qlen, 768), E_d(batch_size, dlen, 768)
    if E_q.dim() != 3:
        print(f"⚠️ Warning: E_q is not 3D! Current shape: {E_q.shape}")
    if E_d.dim() != 3:
        print(f"⚠️ Warning: E_d is not 3D! Current shape: {E_d.shape}")
    # E_q(batch_size, qlen, 768), E_d(batch_size, dlen, 768),
    if E_q.device != device:
        E_q = E_q.to(device)
    if E_d.device != device:
        E_d = E_d.to(device)
    E_q.float()
    E_d.float()
    # print(f'calculate_S_qd_regl_batch E_q:{E_q.shape}, E_d:{E_d.shape}')
    E_q_normalized = torch.nn.functional.normalize(E_q, p=2, dim=2)
    E_d_normalized = torch.nn.functional.normalize(E_d, p=2, dim=2)
    # print(f'calculate_S_qd_regl_batch E_q_normalized:{E_q_normalized.shape}, E_d_normalized:{E_d_normalized.shape}')
    cosine_sim_matrix = torch.matmul(E_q_normalized, E_d_normalized.transpose(1, 2))
    # print(f'calculate_S_qd_regl_batch cosine_sim_matrix:{cosine_sim_matrix.shape}')
    max_scores, _ = torch.max(cosine_sim_matrix, dim=2)
    # print(f'calculate_S_qd_regl_batch max_scores:{max_scores.shape}')
    S_qd_scores = max_scores.sum(dim=1)
    # print(f'calculate_S_qd_regl_batch S_qd_scores:{S_qd_scores.shape}')
    # E_q = E_q.cpu()
    # E_d = E_d.cpu()
    # torch.cuda.empty_cache()
    return S_qd_scores.cpu()  # (batch_size,)


def calculate_S_qd_regl_batch_batch(E_q, E_d, device):
    # E_q(a, qlen, 768), E_d(b, dlen, 768)
    E_q = E_q.to(device).float()
    E_d = E_d.to(device).float()
    E_q_normalized = torch.nn.functional.normalize(E_q, p=2, dim=2)
    E_d_normalized = torch.nn.functional.normalize(E_d, p=2, dim=2)
    # 텐서 확장 (E_q와 E_d의 차원을 맞추기 위해)
    E_q_expanded = E_q_normalized.unsqueeze(1)  # (a, 1, 254, 768)
    E_d_expanded = E_d_normalized.unsqueeze(0)  # (1, b, 4096, 768)

    # 코사인 유사도 계산 (내적)
    cosine_sim_matrix = torch.matmul(
        E_q_expanded, E_d_expanded.transpose(3, 2)
    )  # (a, b, 254, 66556)
    # 각 쿼리와 문서 간의 최대 코사인 유사도
    max_scores, _ = torch.max(cosine_sim_matrix, dim=3)  # (a, b, 254)
    # 최대 코사인 유사도의 합
    S_qd_scores = max_scores.sum(dim=2)  # (a, b)
    # print(f"E_q:{E_q.shape}, E_d:{E_d.shape}, max_scores:{max_scores.shape}, S_qd_score: {S_qd_score}")
    E_q = E_q.cpu()
    E_d = E_d.cpu()
    torch.cuda.empty_cache()
    return S_qd_scores  # (a, b)


# find_best_k_experiment
def calculate_S_qd_regl_dict(E_q, E_d, device):
    if isinstance(E_q, dict):
        E_q = torch.stack(list(E_q.values()), dim=0)
    if isinstance(E_d, dict):
        E_d = torch.stack(list(E_d.values()), dim=0)
    E_q = E_q.to(device).float()
    E_d = E_d.to(device).float()
    # print(f"E_q: {E_q.shape}, E_d: {E_d.shape}")
    E_q_normalized = torch.nn.functional.normalize(
        E_q, p=2, dim=2
    )  # (batch_size, qlen, 768)
    E_d_normalized = torch.nn.functional.normalize(E_d, p=2, dim=1)  # (hlen, 768)
    cosine_sim_matrix = torch.matmul(
        E_q_normalized, E_d_normalized.T
    )  # (batch_size, qlen, hlen)
    # print(f"cosine_sim_matrix: {cosine_sim_matrix.shape}")
    max_scores, _ = torch.max(
        cosine_sim_matrix, dim=2
    )  # q[i]에 대해 최댓값 h[j] 찾기. (batch_size, qlen)
    # print(f"max_scores: {max_scores.shape}")
    S_qd_score = max_scores.sum(dim=1)  # batch[i]에 대해 합 구하기. (batch_size,)
    E_q = E_q.cpu()
    E_d = E_d.cpu()
    torch.cuda.empty_cache()
    return S_qd_score


def calculate_S_qd_regl_logits(E_q, E_d):
    # E_q: (batch_size, qlen, 768), E_d: (batch_size, samples, dlen, 768)
    device = E_q.device
    E_q = E_q.to(device).float()
    E_d = E_d.to(device).float()
    E_q_normalized = torch.nn.functional.normalize(
        E_q, p=2, dim=-1
    )  # (batch_size, qlen, 768)
    E_d_normalized = torch.nn.functional.normalize(
        E_d, p=2, dim=-1
    )  # (batch_size, samples, dlen, 768)
    cosine_sim_matrix = torch.matmul(
        E_q_normalized.unsqueeze(1),  # (batch_size, 1, qlen, 768)
        E_d_normalized.transpose(2, 3),
    )  # (batch_size, samples, 768, dlen)
    # (batch_size, samples, qlen, dlen) → q별 최대 유사도 계산, 합산
    max_scores, _ = torch.max(cosine_sim_matrix, dim=-1)  # (batch_size, samples, qlen)
    S_qd_scores = max_scores.sum(dim=-1)  # (batch_size, samples)
    E_q = E_q.cpu()
    E_d = E_d.cpu()
    torch.cuda.empty_cache()
    return S_qd_scores

def calculate_S_qd_regl_batch_batch_batch(E_q, E_d, device, batch_size: int = 8):
    """
    E_q: (n_q, qlen, hidden)
    E_d: (n_d, dlen, hidden)
    -> S_qd_scores: (n_q, n_d)

    메모리를 아끼기 위해 (n_q, n_d)를 한 번에 계산하지 않고
    (batch_q, batch_d) 블록 단위로 계산.
    """
    import torch

    E_q = E_q.to(device).float()
    E_d = E_d.to(device).float()

    E_q_normalized = torch.nn.functional.normalize(E_q, p=2, dim=2)
    E_d_normalized = torch.nn.functional.normalize(E_d, p=2, dim=2)

    n_q = E_q_normalized.size(0)
    n_d = E_d_normalized.size(0)

    # 결과 행렬은 CPU에 올려두는 게 보통 안전함 (스코어는 그렇게 크지 않으니까)
    S_qd_scores = torch.empty(n_q, n_d, dtype=torch.float32, device="cpu")

    for i in range(0, n_q, batch_size):
        q_batch = E_q_normalized[i : i + batch_size]  # (bq, qlen, h)
        bq = q_batch.size(0)

        # GPU로 올리고
        q_batch = q_batch.to(device)
        q_batch_expanded = q_batch.unsqueeze(1)  # (bq, 1, qlen, h)

        for j in range(0, n_d, batch_size):
            d_batch = E_d_normalized[j : j + batch_size]  # (bd, dlen, h)
            bd = d_batch.size(0)

            d_batch = d_batch.to(device)
            d_batch_expanded = d_batch.unsqueeze(0)  # (1, bd, dlen, h)

            # (bq, bd, qlen, dlen)
            cosine_sim_block = torch.matmul(
                q_batch_expanded, d_batch_expanded.transpose(3, 2)
            )

            # 각 쿼리 토큰에서 문서 토큰 방향으로 max 후 sum
            max_scores_block, _ = torch.max(cosine_sim_block, dim=3)  # (bq, bd, qlen)
            block_scores = max_scores_block.sum(dim=2)  # (bq, bd)

            # CPU에 있는 최종 행렬에 블록 채우기
            S_qd_scores[i : i + bq, j : j + bd] = block_scores.detach().cpu()

            # 메모리 정리
            del d_batch, d_batch_expanded, cosine_sim_block, max_scores_block, block_scores
            torch.cuda.empty_cache()

        del q_batch, q_batch_expanded
        torch.cuda.empty_cache()

    # 원한다면 여기서도 한 번 더 CPU로 확실히 내려줌 (이미 CPU라 의미는 없지만)
    return S_qd_scores
