import torch


def calculate_S_qd_regl(E_q, E_d, device):
    if isinstance(E_q, list):
        E_q = torch.stack(E_q, dim=0)
    if isinstance(E_d, list):
        E_d = torch.stack(E_d, dim=0)
    E_q = E_q.to(device)
    E_d = E_d.to(device)
    E_q_normalized = torch.nn.functional.normalize(E_q, p=2, dim=1)
    E_d_normalized = torch.nn.functional.normalize(E_d, p=2, dim=1)
    cosine_sim_matrix = torch.matmul(E_q_normalized, E_d_normalized.T)
    max_scores, _ = torch.max(cosine_sim_matrix, dim=1)
    S_qd_score = max_scores.sum()
    return S_qd_score


def calculate_S_qd_regl_batch(E_q, E_d, device):
    E_q = E_q.to(device)
    E_d = E_d.to(device)
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
    return S_qd_scores


def calculate_S_qd_regl_dict_batch(E_q, E_d, device):
    if isinstance(E_d, list):
        E_d = torch.stack([torch.stack(list(E_d_d.values())) for E_d_d in E_d], dim=0)
    E_q = E_q.to(device)
    E_d = E_d.to(device)
    print(f"E_q: {E_q.shape}, E_d: {E_d.shape}")
    E_q_normalized = torch.nn.functional.normalize(E_q, p=2, dim=1)
    E_d_normalized = torch.nn.functional.normalize(E_d, p=2, dim=1)
    cosine_sim_matrix = torch.matmul(E_q_normalized, E_d_normalized.T)
    max_scores, _ = torch.max(cosine_sim_matrix, dim=1)
    S_qd_score = max_scores.sum()
    return S_qd_score


def calculate_S_qd_regl_dict(E_q, E_d, device):
    if isinstance(E_q, dict):
        E_q = torch.stack(list(E_q.values()), dim=0)
    if isinstance(E_d, dict):
        E_d = torch.stack(list(E_d.values()), dim=0)
    E_q = E_q.to(device)
    E_d = E_d.to(device)
    E_q_normalized = torch.nn.functional.normalize(E_q, p=2, dim=1)
    E_d_normalized = torch.nn.functional.normalize(E_d, p=2, dim=1)
    cosine_sim_matrix = torch.matmul(E_q_normalized, E_d_normalized.T)
    max_scores, _ = torch.max(cosine_sim_matrix, dim=1)
    S_qd_score = max_scores.sum()
    return S_qd_score
