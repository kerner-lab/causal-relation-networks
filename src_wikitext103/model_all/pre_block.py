import config
import torch
import torch.nn as nn

class PreBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.emb_size)
        self.wpe = nn.Embedding(config.context_window, config.emb_size)

        pos = torch.arange(config.context_window, dtype=torch.int64).view(1, -1)
        self.register_buffer("pos", pos)

        self.initialize_parameters()

    def forward(self, x):
        #  in: (config.batch_size, config.context_window) dtype: int64
        # out: (config.batch_size, config.context_window, config.emb_size)
        tok_emb = self.wte(x)
        pos_emb = self.wpe(self.pos)
        x = tok_emb + pos_emb
        return x

    def initialize_parameters(self):
        nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.wpe.weight, mean=0.0, std=1.0)


if __name__ == "__main__":
    pre_block = PreBlock()
    x = torch.randint(0, config.vocab_size, size=(config.batch_size, config.context_window), dtype=torch.int64)
    p = pre_block(x)
    print("x.shape", x.shape)
    print("p.shape", p.shape)
    print(p.mean(), p.std())

    pre_block_state_dict = pre_block.state_dict()
    print(pre_block_state_dict.keys())
    print(pre_block_state_dict["pos"])
    a = pre_block_state_dict["wte.weight"]
    b = pre_block_state_dict["wpe.weight"]
    print(a.mean(), a.std())
    print(b.mean(), b.std())

# [x] proofreading
