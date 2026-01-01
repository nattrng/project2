import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class PythiaBinaryDataset(Dataset):
    def __init__(self, data_path, seq_len):
        self.seq_len = seq_len
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} not found. Did you combine the shards?")
        # Map file to memory (uint16 fits the ~50k vocab size)
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.total_tokens = len(self.data)
        self.num_samples = (self.total_tokens - 1) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        chunk = self.data[start_idx:end_idx].astype(np.int64)
        return {
            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
            "labels": torch.tensor(chunk[1:], dtype=torch.long)
        }