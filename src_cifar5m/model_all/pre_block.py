import config
import torch
import torch.nn as nn


class PreBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = nn.Linear(12, config.emb_size, bias=False)
        self.cls_tok = nn.Parameter(torch.empty(1, 1, config.emb_size))
        self.pos_emb = nn.Parameter(torch.empty(1, 257, config.emb_size))  # 257 = 256 + 1

        self.initialize_parameters()

    def forward(self, x):
        #  in: (config.batch_size, config.context_window) dtype: int64
        # out: (config.batch_size, config.context_window, config.emb_size)
        # pixel information to embedding
        x = self.fc_1(x)
        # concatenate cls_tok *to the end*
        x = torch.concat((x, self.cls_tok.repeat(len(x), 1, 1)), dim=1)
        # add positional embedding
        x = x + self.pos_emb
        return x

    def initialize_parameters(self):
        # fc_1
        nn.init.normal_(self.fc_1.weight, mean=0.0, std=0.02)
        # cls_tok
        nn.init.normal_(self.cls_tok, mean=0.0, std=1.0)
        # pos_emb
        nn.init.normal_(self.pos_emb, mean=0.0, std=1.0)

# [x] proofreading
