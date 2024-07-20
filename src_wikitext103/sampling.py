import argparse
import time
import tqdm
import wandb
import config
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_all.dataset_wikitext103 import DatasetWikitext103
from model_all.model import Model
from util_all.get_learning_rate import get_learning_rate
from util_all.set_random_seeds import set_random_seeds
from util_all.get_parameter_count import get_parameter_count
from util_all.autoregressive_decoding import autoregressive_decoding


def sample_from_logits(logits, temperature=1.0):
    logits = logits - logits.max()  # numerical stability
    exp_logits = (logits / temperature).exp()
    probabilities = exp_logits / exp_logits.sum()
    probabilities = probabilities.cpu().detach().numpy()
    return np.random.choice(len(probabilities), p=probabilities)


# model
model = Model().cuda()
# performance considerations
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
# load weights
state_dict = torch.load(config.model_load_path)
state_dict_new = dict()
for key in state_dict:
    state_dict_new[key[10:]] = state_dict[key]
model.load_state_dict(state_dict_new)
print("model loaded from {}".format(config.model_load_path))


beginning = "Three years later,"

num_token_condition = len(beginning)
condition = torch.zeros(1, num_token_condition)
for idx in range(num_token_condition):
    condition[0, idx] = ord(beginning[idx])

# create x and p
x = torch.zeros((1, config.context_window), dtype=torch.int64, device="cuda")
# conditioning
x[:, :num_token_condition] = condition


# beginning
print("\n\nStart\n\n")
for idx in range(num_token_condition):
    sample = condition[0, idx]
    print(chr(int(sample)), end="")
# sampling
for idx in range(num_token_condition - 1, config.context_window):
    logits = model(x)[0, idx]  # assuming batch_size = 1
    
    # sample = logits.argmax()
    sample = sample_from_logits(logits, temperature=0.7)
    if idx != (config.context_window - 1):  # if not the last iteration
        x[:, idx + 1] = sample
    print(chr(sample), end="")
print("\n\n")








