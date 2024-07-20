import torch
import numpy as np
from torch.utils.data import Dataset


class DatasetWikitext103(Dataset):
    def __init__(self, dataset_type, batch_size, context_window):
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.context_window = context_window
        self.dataset_size_all = {"train": 522243436, "val": 1108116, "test": 1240962}
        self.dataset_path_all = {"train": "./../datasets/wikitext103/train.mmap",
                                 "val": "./../datasets/wikitext103/val.mmap",
                                 "test": "./../datasets/wikitext103/test.mmap"}
        self.n_ids = self.dataset_size_all[self.dataset_type]
        self.n_sample = (self.n_ids - 1) // self.context_window  # minus one: due to y steps one wrt x
        self.n_batch = self.n_sample // self.batch_size  # drop last
        self.text = np.memmap(self.dataset_path_all[self.dataset_type], dtype="uint8", mode="r", shape=(self.n_ids,))

        self.map_batch = None
        self.shuffle()

    def shuffle(self):
        self.map_batch = np.arange(self.n_sample, dtype="int64")
        np.random.shuffle(self.map_batch)
        self.map_batch = self.map_batch[:self.n_batch * self.batch_size]
        self.map_batch = self.map_batch.reshape((self.n_batch, self.batch_size))

    def __len__(self):
        return self.n_batch

    def __getitem__(self, idx_batch):
        batch_x = torch.zeros((self.batch_size, self.context_window), dtype=torch.int64)
        batch_y = torch.zeros((self.batch_size, self.context_window), dtype=torch.int64)

        for idx_within_batch in range(self.batch_size):
            idx_within_text = self.map_batch[idx_batch, idx_within_batch]
            idx_start = idx_within_text * self.context_window
            idx_end = idx_start + self.context_window + 1
            x_and_y = self.text[idx_start:idx_end]
            x_and_y = torch.from_numpy(x_and_y.astype("int64"))

            batch_x[idx_within_batch] = x_and_y[:-1]
            batch_y[idx_within_batch] = x_and_y[1:]
        return batch_x, batch_y

# [x] proofreading
# [x] testing