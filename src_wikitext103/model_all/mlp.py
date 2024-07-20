import math
import config
import torch.nn as nn
from model_all.normalize import normalize


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = nn.Linear(config.emb_size, config.hid_size, bias=True)
        self.fc_2 = nn.Linear(config.hid_size, config.emb_size, bias=True)
        self.gelu = nn.GELU()

        self.initialize_parameters()

    def forward(self, x):
        x = normalize(x)
        x = self.fc_1(x)
        x = self.gelu(x)
        x = self.fc_2(x)
        return x

    def initialize_parameters(self):
        nn.init.normal_(self.fc_1.weight, mean=0, std=0.02)
        nn.init.zeros_(self.fc_1.bias)
        nn.init.normal_(self.fc_2.weight, mean=0, std=0.02 / math.sqrt(2 * config.num_block))
        nn.init.zeros_(self.fc_2.bias)


if __name__ == "__main__":
    mlp = MLP().cuda()
    mlp_state_dict = mlp.state_dict()
    a = mlp_state_dict["fc_1.weight"]
    b = mlp_state_dict["fc_1.bias"]
    c = mlp_state_dict["fc_2.weight"]
    d = mlp_state_dict["fc_2.bias"]
    print(a.mean(), a.std())
    print(b.mean(), b.std())
    print(c.mean(), c.std())
    print(d.mean(), d.std())
    
    import torch
    x = torch.randn(config.batch_size, config.context_window, config.emb_size, device="cuda")
    p = mlp(x)
    

# checked
