import torch
import numpy as np
from torch.utils.data import Dataset

class DatasetCifar5m(Dataset):
    def __init__(self, batch_size):
        self.images = np.memmap("./../datasets/cifar5m/images.mmap", dtype="uint8", mode="r", shape=(6002688, 32, 32, 3))
        self.labels = np.memmap("./../datasets/cifar5m/labels.mmap", dtype="int64", mode="r", shape=(6002688,))
        self.batch_size = batch_size

        # Prepare mean and std
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 1, 3)
        self.std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 1, 3)

        # get num_batch and map_batch
        num_sample = len(self.images)
        num_batch = num_sample // batch_size  # drop last
        map_batch = np.arange(num_sample, dtype="int64")
        np.random.shuffle(map_batch)
        map_batch = map_batch[:num_batch * batch_size]
        map_batch = map_batch.reshape((num_batch, batch_size))
        self.num_batch = num_batch
        self.map_batch = map_batch

    def __len__(self):
        return self.num_batch

    def __getitem__(self, idx_batch):
        # Initialize batch_image and batch_label
        batch_image = torch.zeros((self.batch_size, 256, 12), dtype=torch.float32)
        batch_label = torch.zeros((self.batch_size,), dtype=torch.int64)

        for idx_within_batch in range(self.batch_size):
            # Get image and label
            idx_within_images = self.map_batch[idx_batch, idx_within_batch]
            image = self.images[idx_within_images]
            label = self.labels[idx_within_images]

            # To torch tensors
            image = torch.tensor(image, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.int64)

            # Standardize
            image /= 255
            image -= self.mean
            image /= self.std

            # Patch
            image = image.view(16, 2, 16, 2, 3)
            image = image.permute(0, 2, 1, 3, 4)
            image = image.contiguous()
            image = image.view(256, 12)

            # Store image and label in batch_image and batch_label
            batch_image[idx_within_batch] = image
            batch_label[idx_within_batch] = label
        return batch_image, batch_label

# [x] proofreading
