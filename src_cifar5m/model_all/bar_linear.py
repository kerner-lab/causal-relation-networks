import config
import torch
import torch.nn as nn
from model_all.normalize import normalize

denominators_causal = torch.arange(config.context_window)[None, :, None].float().cuda() + 1.0

# checked
def bar_linear_bidirectional(a, b):
    return a.exp() * b.exp().mean(dim=1, keepdim=True)

# checked
def bar_linear_bidirectional_stable(a, b):
    b = b.logsumexp(dim=1, keepdim=True)
    const_b = b.max(dim=1, keepdim=True)[0].detach()  # second invariance
    b = b - const_b
    a = a + const_b
    const_a = a.amax(dim=2, keepdim=True).detach()  # first invariance
    a = a - const_a
    return a.exp() * b.exp()

# checked
def bar_linear_causal(a, b):
    return a.exp() * b.exp().cumsum(dim=1) / denominators_causal

# checked
def bar_linear_causal_stable(a, b):
    b = b.logcumsumexp(dim=1)
    const_b = b.cummax(dim=1)[0].detach()  # second invariance
    b = b - const_b
    a = a + const_b
    const_a = a.amax(dim=2, keepdim=True).detach()  # first invariance
    a = a - const_a
    return a.exp() * b.exp() / denominators_causal

if __name__ == "__main__":
    a = torch.randn(config.batch_size, config.context_window, config.hid_size, dtype=torch.float64).cuda()
    b = torch.randn(config.batch_size, config.context_window, config.hid_size, dtype=torch.float64).cuda()
    c = bar_linear_causal(a, b)
    print(a.shape)
    print(b.shape)
    print(c.shape)

# [x] testing
