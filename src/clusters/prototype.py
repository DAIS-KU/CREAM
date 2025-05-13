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

    def _get_key(self, embeddings, device):
        random_vectors = self.random_vectors.to(device)
        projections = torch.matmul(embeddings, random_vectors.T)
        binary_vectors = (projections > 0).int()
        int_keys = torch.sum(binary_vectors * self.powers_of_two.to(device), dim=1)
        return int_keys.tolist()

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
        if self.use_tensor_key:
            compressed_embeddings = torch.zeros(self.hash_size, self.embedding_dim)
            for key, values in hash_table.items():
                values_tensor = torch.stack(values)
                compressed_embeddings[key] = values_tensor.sum(dim=0)
            return compressed_embeddings.cpu()
        else:
            sparse_map = defaultdict(int)
            for key, values in hash_table.items():
                sparse_map[key] = torch.stack(values).sum(dim=0).cpu()
            return sparse_map

    def encode(self, embeddings):
        device = embeddings.device
        hash_table = self._hash(embeddings, device)
        final_vector = self.get_final_vector(hash_table)
        return final_vector
