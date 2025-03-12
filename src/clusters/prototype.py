from collections import defaultdict

import torch


class RandomProjectionLSH:
    def __init__(self, random_vectors, embedding_dim=768):
        self.embedding_dim = embedding_dim
        self.random_vectors = random_vectors.clone()

    def _get_key(self, embeddings, device):
        random_vectors = self.random_vectors.to(device)
        projections = torch.matmul(embeddings, random_vectors.T)
        binary_vectors = (projections > 0).int()
        powers_of_two = 2 ** torch.arange(binary_vectors.size(1), device=device).flip(0)
        int_keys = torch.sum(binary_vectors * powers_of_two, dim=1)
        return int_keys.tolist()

    def _hash(self, embeddings, device):
        # print(f'embeddings :{embeddings.shape}')
        valid_mask = ~torch.all(embeddings == 0, dim=1)
        valid_embeddings = embeddings[valid_mask]
        hash_keys = self._get_key(valid_embeddings, device)
        hash_table = defaultdict(list)
        for idx, key in enumerate(hash_keys):
            hash_table[key].append(valid_embeddings[idx])
        return hash_table

    def get_final_vector(self, hash_table):
        sparse_map = defaultdict(int)
        for key, values in hash_table.items():
            sparse_map[key] = torch.stack(values).sum(dim=0).cpu()
        return sparse_map

    def encode(self, embeddings):
        device = embeddings.device
        hash_table = self._hash(embeddings, device)
        sparse_map = self.get_final_vector(hash_table)
        return sparse_map
