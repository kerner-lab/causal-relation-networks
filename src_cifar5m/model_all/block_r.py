import math
import config
import torch.nn as nn
from model_all.bar_quadratic import bar_quadratic_causal_preact_stable
from model_all.normalize import normalize

# note: this is always causal
class BlockR(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = nn.Linear(config.emb_size, config.hid_size, bias=config.enable_relation_network_bias_1)
        self.fc_2 = nn.Linear(config.emb_size, config.hid_size, bias=config.enable_relation_network_bias_2)
        self.fc_3 = nn.Linear(config.hid_size, config.emb_size, bias=config.enable_relation_network_bias_3)

        self.initialize_parameters()

    def forward(self, x):
        r = x
        r = normalize(r)
        a = self.fc_1(r)
        b = self.fc_2(r)
        r = bar_quadratic_causal_preact_stable(a, b)
        r = normalize(r)
        r = self.fc_3(r)
        x = x + r
        return x

    def initialize_parameters(self):
        nn.init.normal_(self.fc_1.weight, mean=0, std=0.02)
        if config.enable_relation_network_bias_1:
            nn.init.zeros_(self.fc_1.bias)
        nn.init.normal_(self.fc_2.weight, mean=0, std=0.02)
        if config.enable_relation_network_bias_2:
            nn.init.zeros_(self.fc_2.bias)
        nn.init.normal_(self.fc_3.weight, mean=0, std=0.02 / math.sqrt(config.num_block))
        if config.enable_relation_network_bias_3:
            nn.init.zeros_(self.fc_3.bias)


if __name__ == "__main__":
    import torch
    block = BlockR().cuda()
    x = torch.randn((config.batch_size, config.context_window, config.emb_size), device="cuda")
    p = block(x)
    print(x.mean().item(), x.std().item())
    print(p.mean().item(), p.std().item())
