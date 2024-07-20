import tqdm
import torch
import config

def autoregressive_decoding(model, condition):
    # note: condition shape: (batch_size, num_token)
    # get batch_size
    batch_size = len(condition)
    # preparations
    condition = condition.clone()  # ask: is this necessary? (to avoid side effects)
    num_token_condition = condition.shape[1]
    # create x and p
    x = torch.zeros((batch_size, config.context_window), dtype=torch.int64, device="cuda")
    p = -torch.ones((batch_size, config.context_window), dtype=torch.int64, device="cuda")
    # conditioning
    x[:, :num_token_condition] = condition
    # sampling
    for idx in tqdm.tqdm(range(num_token_condition - 1, config.context_window)):
        sample = model(x).argmax(dim=-1)[:, idx]
        p[:, idx] = sample
        if idx != (config.context_window - 1):  # if not the last iteration
            x[:, idx + 1] = sample
    return p

# checked
