import time
import tqdm
import wandb
import config
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from util_all.get_learning_rate import get_learning_rate
from dataset_all.dataset_wikitext103 import DatasetWikitext103
from model_all.model import Model
from util_all.set_random_seeds import set_random_seeds
from util_all.get_parameter_count import get_parameter_count

# set random seed
set_random_seeds()

# wandb
run = wandb.init(project=config.wandb_project_name, name=config.wandb_run_name)

# dataset
if config.enable_hyperparameter_tuning_mode:
    dataset = DatasetWikitext103("val", config.batch_size * config.num_grad_accu, config.context_window)  # debug: should be val not train
else:
    dataset = DatasetWikitext103("train", config.batch_size * config.num_grad_accu, config.context_window)
dataloader = DataLoader(dataset, batch_size=None, num_workers=config.num_worker_dataloader,
                        pin_memory=True)  # debug: pros and cons of pin_memory?
# model
model = Model().cuda()
# report parameter count
get_parameter_count(model)

# performance considerations
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
if config.enable_torch_compile:
    # model = torch.compile(model)
    model = torch.compile(model, fullgraph=True, mode="max-autotune-no-cudagraphs")
    

# optimization
criterion = nn.CrossEntropyLoss(ignore_index=-1)  # positions with tok_id=-1 are excluded from loss calculation
optimizer = torch.optim.AdamW(params=model.parameters(),
                              betas=(config.adamw_beta_1, config.adamw_beta_2),
                              eps=config.adamw_eps,
                              weight_decay=config.adamw_weight_decay)

# training
idx_iter = 0
model.train()
for inputs, targets in tqdm.tqdm(dataloader):
    torch.cuda.synchronize()
    t1 = time.time()
    # process data
    inputs_chunked = torch.chunk(inputs, config.num_grad_accu)
    targets_chunked = torch.chunk(targets, config.num_grad_accu)
    # zero_grad
    for param in model.parameters():
        param.grad = None
    # set lr
    lr = get_learning_rate(idx_iter)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    # gradient accumulation
    loss_log = 0
    # other
    accuracy_all_soft = []
    for idx_grad_accu in range(config.num_grad_accu):
        # load data
        inputs_current = inputs_chunked[idx_grad_accu].cuda()
        targets_current = targets_chunked[idx_grad_accu].cuda()
        # forward
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(inputs_current)
            loss = criterion(outputs.view(-1, config.vocab_size), targets_current.view(-1))
        # (evaluation) calculate (soft) accuracy; assumption: model.train() is the same as model.eval()
        pass  # only used by copying and CIFAR10
        # normalize the loss to account for accumulation
        loss /= config.num_grad_accu
        # record loss for log
        loss_log += loss.item()
        # backward
        loss.backward()
    # gradient clipping
    if config.enable_grad_clip:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_max_norm)
    # step
    optimizer.step()
    torch.cuda.synchronize()
    t2 = time.time()
    
    # get iter_time
    if idx_iter == 0:
        iter_time = None  # ignore compile time
    else:
        iter_time = t2 - t1

    # to wandb
    run.log({"train_loss": loss_log, "learning_rate": lr, "soft_accuracy": None, "iter_time": iter_time})
    # other
    idx_iter += 1


# save model
torch.save(model.state_dict(), config.model_save_path)
print("model saved to:", config.model_save_path)

# finish the wandb run after training is complete
run.finish()

# [ ] proofread