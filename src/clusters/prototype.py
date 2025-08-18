from collections import defaultdict

import torch


class RandomProjectionLSH:
    def __init__(self, random_vectors, embedding_dim=768, use_tensor_key=False):
        self.embedding_dim = embedding_dim
        self.random_vectors = random_vectors.clone()
        self.num_bits = random_vectors.shape[0]
        self.powers_of_two = 2 ** torch.arange(
            self.num_bits, device=random_vectors.device
        ).flip(0)
        self.hash_size = 1 << self.num_bits
        self.use_tensor_key = use_tensor_key
        print(f"RandomProjectionLSH use_tensor_key {use_tensor_key}")

    def _get_key(self, embeddings, device, is_list=True):
        random_vectors = self.random_vectors.to(device)
        projections = torch.matmul(embeddings, random_vectors.T)
        binary_vectors = (projections > 0).int()
        int_keys = torch.sum(binary_vectors * self.powers_of_two.to(device), dim=1)
        key= int_keys.tolist() if is_list else int(int_keys.cpu())
        # print(f"Generated keys: {key}")
        return key

    def _hash(self, embeddings, device):
        # print(f'embeddings :{embeddings.shape}')
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)
        valid_mask = ~torch.all(embeddings == 0, dim=1)
        valid_embeddings = embeddings[valid_mask]
        hash_keys = self._get_key(valid_embeddings, device)
        hash_table = defaultdict(list)
        for idx, key in enumerate(hash_keys):
            hash_table[key].append(valid_embeddings[idx])
        return hash_table

    def get_final_vector(self, hash_table):
        # if self.use_tensor_key:
        compressed_embeddings = torch.zeros(self.hash_size, self.embedding_dim)
        for key, values in hash_table.items():
            values_tensor = torch.stack(values)
            compressed_embeddings[key] = values_tensor.sum(dim=0)
        return compressed_embeddings.cpu()
        # else:
        #     sparse_map = defaultdict(int)
        #     for key, values in hash_table.items():
        #         sparse_map[key] = torch.stack(values).sum(dim=0).cpu()
        #     return sparse_map

    def encode(self, embeddings):
        device = embeddings.device
        hash_table = self._hash(embeddings, device)
        final_vector = self.get_final_vector(hash_table)
        return final_vector

    def encode_batch(self, batch_embeddings, is_sum=True):
        """
        Args:
            batch_embeddings: Tensor of shape (B, L, D)
        Returns:
            compressed_embeddings: Tensor of shape (hash_size, D)
        """
        B, L, D = batch_embeddings.shape
        device = batch_embeddings.device
        R = self.random_vectors.to(device)  # (num_bits, D)
        projections = torch.matmul(batch_embeddings, R.T)  # (B, L, num_bits)
        binary_vectors = (projections > 0).int()  # (B, L, num_bits)
        hash_keys = torch.sum(
            binary_vectors * self.powers_of_two.to(device), dim=-1
        )  # (B, L), 2진수 결과를 10진수 키로.
        valid_mask = ~(batch_embeddings == 0).all(dim=-1)  # (B, L)
        hash_keys = hash_keys * valid_mask  # 무효 위치를 0으로
        compressed_embeddings = torch.zeros(B, self.hash_size, D, device=device)
        hash_keys_exp = hash_keys.unsqueeze(-1).expand(-1, -1, D)  # (B, L, D)
        compressed_embeddings.scatter_add_(
            dim=1, index=hash_keys_exp, src=batch_embeddings
        )
        if is_sum:
            # 모든 배치의 해시 결과를 합산 → (hash_size, D)
            compressed_sum = compressed_embeddings.sum(dim=0)  # (hash_size, D)
            return compressed_sum.cpu()
        else:
            return compressed_embeddings.cpu()  # (B, hash_size, D)
