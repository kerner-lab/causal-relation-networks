import config
import torch.nn as nn
from model_all.normalize import normalize


class PostBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_cls = nn.Linear(config.emb_size, config.num_class, bias=False)

        self.initialize_parameters()

    def forward(self, x):
        #  in: (batch_size, context_window, emb_size)
        # out: (batch_size, num_class)
        x = x[:, -1]
        x = normalize(x)
        x = self.fc_cls(x)
        return x

    def initialize_parameters(self):
        nn.init.normal_(self.fc_cls.weight, mean=0, std=0.02)

# [x] proofreading
