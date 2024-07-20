import config
import torch.nn as nn
from model_all.pre_block import PreBlock
from model_all.block_t import BlockT
from model_all.block_t_linear import BlockTLinear
from model_all.block_r import BlockR
from model_all.block_r_linear import BlockRLinear
from model_all.block_r_ablation import BlockRAblation

from model_all.post_block import PostBlock


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_block = PreBlock()
        if config.model_name == "Transformer":
            self.block_all = nn.ModuleList([BlockT() for _ in range(config.num_block)])
        elif config.model_name == "Linear Transformer":
            self.block_all = nn.ModuleList([BlockTLinear() for _ in range(config.num_block)])
        elif config.model_name == "Relation Network":
            self.block_all = nn.ModuleList([BlockR() for _ in range(config.num_block)])
        elif config.model_name == "Linear Relation Network":
            self.block_all = nn.ModuleList([BlockRLinear() for _ in range(config.num_block)])
        elif config.model_name == "Relation Network Ablation":
            self.block_all = nn.ModuleList([BlockRAblation() for _ in range(config.num_block)])
        else:
            raise Exception()
        self.post_block = PostBlock()

    def forward(self, x):
        x = self.pre_block(x)
        for block in self.block_all:
            x = block(x)
        x = self.post_block(x)
        return x

# [x] proofreading
