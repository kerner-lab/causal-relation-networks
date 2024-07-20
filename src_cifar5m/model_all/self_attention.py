import math
import config
import torch.nn as nn
import torch.nn.functional as F
from model_all.normalize import normalize


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = nn.Linear(config.emb_size, 3 * config.emb_size, bias=False)
        self.fc_2 = nn.Linear(config.emb_size, config.emb_size, bias=False)

        self.initialize_parameters()

    def forward(self, x):
        #  in shape: (batch_size, context_window, emb_size)
        # out shape: (batch_size, context_window, emb_size)
        x = normalize(x)
        B, T, C = x.size()
        q, k, v = self.fc_1(x).split(config.emb_size, dim=2)
        k = k.view(B, T, config.num_head, config.head_size).transpose(1, 2)
        q = q.view(B, T, config.num_head, config.head_size).transpose(1, 2)
        v = v.view(B, T, config.num_head, config.head_size).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=config.enable_causal_mode)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.fc_2(y)
        return y

    def initialize_parameters(self):
        nn.init.normal_(self.fc_1.weight, mean=0, std=0.02)
        nn.init.normal_(self.fc_2.weight, mean=0, std=0.02 / math.sqrt(2 * config.num_block))

if __name__ == "__main__":
    tmp = SelfAttention()
    tmp_state_dict = tmp.state_dict()
    a = tmp_state_dict["fc_1.weight"]
    b = tmp_state_dict["fc_2.weight"]
    print(a.mean())
    print(b.mean())
    print(a.std(), "vs", 0.0)
    print(b.std(), "vs", 0.02 / math.sqrt(2 * config.num_block))
    print("config.num_block", config.num_block)

# checked
