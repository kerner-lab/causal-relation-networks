import math
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_all.normalize import normalize
from model_all.self_attention import SelfAttention


class LinearSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = nn.Linear(config.emb_size, 3 * config.emb_size, bias=False)
        self.fc_2 = nn.Linear(config.emb_size, config.emb_size, bias=False)

        self.initialize_parameters()
        
        causal_mask = torch.tril(torch.ones(config.context_window, config.context_window))
        causal_mask = causal_mask.view(1, 1, config.context_window, config.context_window)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        #  in shape: (batch_size, context_window, emb_size)
        # out shape: (batch_size, context_window, emb_size)
        x = normalize(x)
        B, T, C = x.size()
        q, k, v = self.fc_1(x).split(config.emb_size, dim=2)
        k = k.view(B, T, config.num_head, config.head_size).transpose(1, 2)
        q = q.view(B, T, config.num_head, config.head_size).transpose(1, 2)
        v = v.view(B, T, config.num_head, config.head_size).transpose(1, 2)
        # ----- #
        # manual implementation of (linear) self-attention
        # ----- #
        q = torch.nn.functional.elu(q) + 1
        k = torch.nn.functional.elu(k) + 1
        att = q @ k.transpose(-2, -1)        
        # att = att / math.sqrt(config.emb_size)  # doesn't matter for linear self-attention
        # att = att.exp()  # for quadratic self-attention
        if config.enable_causal_mode:
            att = att.masked_fill(self.causal_mask == 0, 0.0)  # apply the causal mask
        att = att / att.sum(dim=-1, keepdim=True)
        y = att @ v
        # ----- #
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.fc_2(y)
        return y

    def initialize_parameters(self):
        nn.init.normal_(self.fc_1.weight, mean=0, std=0.02)
        nn.init.normal_(self.fc_2.weight, mean=0, std=0.02 / math.sqrt(2 * config.num_block))

if __name__ == "__main__":
    config.context_window = 3
    torch.manual_seed(42)
    model_1 = SelfAttention().cuda()
    torch.manual_seed(42)
    model_2 = LinearSelfAttention().cuda()
    a = torch.randn(config.batch_size, config.context_window, config.emb_size, dtype=torch.float32, device="cuda")
    out_1 = model_1(a)
    out_2 = model_2(a)
    print((out_1 - out_2).abs().max())

# checked
