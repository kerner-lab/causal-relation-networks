import time
import tqdm
import wandb
import config
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_all.dataset_cifar5m import DatasetCifar5m
from model_all.model import Model
from model_all.normalize import normalize
from util_all.get_learning_rate import get_learning_rate
from util_all.set_random_seeds import set_random_seeds
from util_all.get_parameter_count import get_parameter_count

idx_to_class = ["airplane", "automobile", "bird", "cat",
                "deer", "dog", "frog", "horse", "ship", "truck"]
set_random_seeds()

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


dataset = DatasetCifar5m(config.num_grad_accu * config.batch_size)

idx_sample = 0
x, y = dataset[idx_sample]

with torch.no_grad():
    # inference
    p_all = []
    p_all.append(model.pre_block(x.cuda()))
    for idx in range(12):
        p_all.append(model.block_all[idx](p_all[idx]))
    p = model.post_block(p_all[-1])

    # get_heatmap
    idx_heatmap = 256
    idx_block = 5
    module_current = model.block_all[idx_block]
    r = p_all[idx_block]
    a = module_current.fc_1(r)
    b = module_current.fc_2(r)
    a = normalize(a)
    b = normalize(b)
    a = a.view(1, 257, 1, 3072)
    b = b.view(1, 1, 257, 3072)
    # actual computation
    out = a + b  # broadcast
    out = out.exp()  # activation
    heatmap_current = out[0, idx_heatmap]
    heatmap_current = heatmap_current[:-1]  # remove <CLS>
    heatmap_current = heatmap_current.norm(dim=-1)
    heatmap_current = heatmap_current / heatmap_current.sum()
    heatmap_current = heatmap_current.reshape(16, 16)
    plt.imshow(heatmap_current.cpu(), cmap="hot")
    plt.colorbar(orientation='vertical')
    plt.axis("off")
    plt.savefig("./{}_heatmap.png".format(idx_sample))
    plt.show()


idx_current = 0
x_current = x[idx_current]
y_current = y[idx_current]
p_current = p[idx_current]

x_current = x_current.view(16, 16, 2, 2, 3)
x_current = x_current.permute(0, 2, 1, 3, 4).contiguous()
x_current = x_current.view(32, 32, 3)

x_current = x_current * dataset.std
x_current = x_current + dataset.mean
x_current = x_current * 255

x_current = x_current.numpy().astype("uint8")

plt.imshow(x_current)
plt.axis("off")
plt.savefig("./{}_img.png".format(idx_sample))
plt.show()


print("predicted class:", idx_to_class[p_current.argmax()])
print("   actual class:", idx_to_class[y_current])




