import config
import torch
import torch.nn as nn
from model_all.normalize import normalize

denominators_causal = torch.arange(config.context_window)[None, :, None].float().cuda() + 1.0
u = torch.arange(config.context_window, device="cuda").view(1, config.context_window, 1, 1)
v = torch.arange(config.context_window, device="cuda").view(1, 1, config.context_window, 1)
causal_mask = u >= v
        
        
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
    out = out.exp()  # activation
    out = out.sum(dim=2)  # reduction
    out = out / denominators_causal  # averaging
    # tensor manipulation
    out = out.view(-1, config.num_head, config.context_window, config.head_size)
    out = out.permute(0, 2, 1, 3).contiguous()
    out = out.view(-1, config.context_window, config.num_head * config.head_size)
    return out

# checked
def bar_quadratic_causal_preact_stable(a, b):
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
    out = out.logsumexp(dim=2)  # activation and reduction
    out = out - out.amax(dim=2, keepdim=True).detach()  # stability
    out = out.exp()
    out = out / denominators_causal  # averaging
    # tensor manipulation
    out = out.view(-1, config.num_head, config.context_window, config.head_size)
    out = out.permute(0, 2, 1, 3).contiguous()
    out = out.view(-1, config.context_window, config.num_head * config.head_size)
    return out

if __name__ == "__main__":
    from model_all.bar import BAR
    bar_ori = BAR()
    
    a = torch.randn(config.batch_size, config.context_window, config.hid_size, dtype=torch.float32).cuda()
    b = torch.randn(config.batch_size, config.context_window, config.hid_size, dtype=torch.float32).cuda()    
    c = bar_quadratic_causal_preact(a, b)
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(c.mean(dim=-1).mean(0))

# [x] testing
