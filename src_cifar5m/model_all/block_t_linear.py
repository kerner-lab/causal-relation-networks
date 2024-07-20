import torch.nn as nn
from model_all.linear_self_attention import LinearSelfAttention
from model_all.mlp import MLP


class BlockTLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = LinearSelfAttention()
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


if __name__ == "__main__":
    import torch
    import config
    block = BlockTLinear().cuda()
    x = torch.randn((config.batch_size, config.context_window, config.emb_size), device="cuda")
    p = block(x)
    print(x)
    print(p)
    print(x.mean(), x.std())
    print(p.mean(), p.std())
