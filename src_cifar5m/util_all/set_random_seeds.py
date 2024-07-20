import torch
import random
import numpy as np
import config


def set_random_seeds():
    if config.random_seed is not None:
        random.seed(config.random_seed)
        np.random.seed(config.random_seed + 1)
        torch.manual_seed(config.random_seed + 2)


if __name__ == "__main__":
    set_random_seeds()
    print(torch.rand(1).item())
    print(np.random.random())
    print(random.random())
    set_random_seeds()
    print(torch.rand(1).item())
    print(np.random.random())
    print(random.random())

# checked
