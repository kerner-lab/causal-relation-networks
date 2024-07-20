import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_all.normalize import normalize

denominators_causal = torch.arange(config.context_window)[None, :, None].float().cuda() + 1.0
u = torch.arange(config.context_window, device="cuda").view(1, config.context_window, 1, 1)
v = torch.arange(config.context_window, device="cuda").view(1, 1, config.context_window, 1)
causal_mask = u >= v
gelu = nn.GELU()
elu = nn.ELU()

# checked
def bar_quadratic_causal_preact(a, b):
    # tensor manipulation
    a = a.view(-1, config.context_window, 1, config.num_head, config.head_size)
    b = b.view(-1, 1, config.context_window, config.num_head, config.head_size)
    a = a.permute(0, 3, 1, 2, 4).contiguous()
    b = b.permute(0, 3, 1, 2, 4).contiguous()
    a = a.view(-1, config.context_window, 1, config.head_size)
    b = b.view(-1, 1, config.context_window, config.head_size)
    # actual computation
    out = a + b  # broadcast
    out = normalize(out)  # normalization
    out = out.masked_fill(causal_mask == 0, float("-inf"))  # apply the causal mask
    # ----- #
    # activation
    # ----- #
    if config.ablation_parameters["Activation"] == "exp":
        out = out.exp()
    elif config.ablation_parameters["Activation"] == "gelu":
        out = gelu(out)
    elif config.ablation_parameters["Activation"] == "elu":
        out = elu(out)
    elif config.ablation_parameters["Activation"] == "relu":
        out = out.relu()
    else:
        raise Exception()
    # ----- #
    out = out.sum(dim=2)  # reduction
    out = out / denominators_causal  # averaging
    # tensor manipulation
    out = out.view(-1, config.num_head, config.context_window, config.head_size)
    out = out.permute(0, 2, 1, 3).contiguous()
    out = out.view(-1, config.context_window, config.num_head * config.head_size)
    return out

# checked
def bar_quadratic_causal(a, b):
    # tensor manipulation
    a = a.view(-1, config.context_window, 1, config.num_head, config.head_size)
    b = b.view(-1, 1, config.context_window, config.num_head, config.head_size)
    a = a.permute(0, 3, 1, 2, 4).contiguous()
    b = b.permute(0, 3, 1, 2, 4).contiguous()
    a = a.view(-1, config.context_window, 1, config.head_size)
    b = b.view(-1, 1, config.context_window, config.head_size)
    # actual computation
    out = a + b  # broadcast
    out = out.masked_fill(causal_mask == 0, float("-inf"))  # apply the causal mask
    
    # ----- #
    # activation
    # ----- #
    if config.ablation_parameters["Activation"] == "exp":
        out = out.exp()
    elif config.ablation_parameters["Activation"] == "gelu":
        out = gelu(out)
    elif config.ablation_parameters["Activation"] == "elu":
        out = F.elu(out)
    elif config.ablation_parameters["Activation"] == "relu":
        out = out.relu()
    else:
        raise Exception()
    # ----- #
    
    out = out.sum(dim=2)  # reduction
    out = out / denominators_causal  # averaging
    # tensor manipulation
    out = out.view(-1, config.num_head, config.context_window, config.head_size)
    out = out.permute(0, 2, 1, 3).contiguous()
    out = out.view(-1, config.context_window, config.num_head * config.head_size)
    return out
    

# checked
