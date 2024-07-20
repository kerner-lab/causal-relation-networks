import torch.nn as nn
from model_all.self_attention import SelfAttention
from model_all.mlp import MLP


class BlockT(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = SelfAttention()
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


if __name__ == "__main__":
    import torch
    import config
    block = BlockT().cuda()
    x = torch.randn((config.batch_size, config.context_window, config.emb_size), device="cuda")
    p = block(x)
    print(x)
    print(p)
    print(x.mean(), x.std())
    print(p.mean(), p.std())
