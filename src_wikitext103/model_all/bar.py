import config
import torch
import torch.nn as nn
from model_all.normalize import normalize

# assumption: this is causal
class BAR(nn.Module):
    def __init__(self):
        super().__init__()
        # get causal_mask
        u = torch.arange(config.context_window, device="cuda").view(1, config.context_window, 1, 1)
        v = torch.arange(config.context_window, device="cuda").view(1, 1, config.context_window, 1)
        self.causal_mask = u >= v

    def forward(self, a, b):
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
        out = out.exp()  # activation
        out = out * self.causal_mask  # causal masking
        out = out.sum(dim=2)  # reduction
        # tensor manipulation
        out = out.view(-1, config.num_head, config.context_window, config.head_size)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(-1, config.context_window, config.num_head * config.head_size)
        return out

if __name__ == "__main__":
    bar = BAR()
    a = torch.randn(config.batch_size, config.context_window, config.hid_size).cuda()
    b = torch.randn(config.batch_size, config.context_window, config.hid_size).cuda()
    c = bar(a, b)
    print("a shape: ", a.shape)
    print("b shape: ", b.shape)
    print("c shape: ", c.shape)

# [x] proofreading
