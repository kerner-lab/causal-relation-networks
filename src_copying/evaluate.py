import tqdm
import wandb
import config
import torch
import torch.nn as nn
from util_all.get_learning_rate import get_learning_rate
from dataset_all.dataset_copying import DatasetCopying
from model_all.model import Model
from itertools import islice
from util_all.autoregressive_decoding import autoregressive_decoding
from util_all.calculate_accuracy_copying import calculate_accuracy_copying


def evaluate(dataloader, model, soft_only=False):
    # debug: torch no grad
    model.eval()
    accuracy_all_soft = []
    accuracy_all_hard = []
    for x, y in islice(dataloader, config.num_iter_eval):
        x = x.cuda()  # (batch_size, num_token)
        y = y.cuda()  # (batch_size, num_token)
        # calculate soft accuracy
        p_soft = model(x).argmax(dim=-1)
        accuracy_all_soft.append(calculate_accuracy_copying(p_soft, y).tolist())
        if not soft_only:
            # calculate hard accuracy
            p_hard= autoregressive_decoding(model, condition=x[:, :config.string_size + 2])
            accuracy_all_hard.append(calculate_accuracy_copying(p_hard, y).tolist())
    accuracy_all_soft = torch.tensor(accuracy_all_soft, dtype=torch.float32)
    accuracy_all_hard = torch.tensor(accuracy_all_hard, dtype=torch.float32)
    return accuracy_all_soft, accuracy_all_hard

# [ ] proofread
