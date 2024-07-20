import math
import config


def get_learning_rate(idx_iter):
    lr_max, lr_terminal, idx_end_warm_up, idx_end_decay = config.lr_schedule_param
    if idx_iter < idx_end_warm_up:
        return lr_max * idx_iter / idx_end_warm_up  # note: idx_end_warm_up can be zero
    elif idx_iter < idx_end_decay:
        ratio = (idx_iter - idx_end_warm_up) / (idx_end_decay - idx_end_warm_up)
        coeff = (math.cos(math.pi * ratio) + 1) / 2
        return lr_terminal + coeff * (lr_max - lr_terminal)
    else:
        return lr_terminal


if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    # config.lr_schedule_param = (1e-4, 1e-4, 0, 1000)
    print("config.lr_schedule_param:", config.lr_schedule_param)
    x_all = torch.arange(4000)
    y_all = torch.zeros(4000)
    for idx in range(len(x_all)):
        y_all[idx] = get_learning_rate(x_all[idx])
    plt.plot(x_all, y_all)

# checked
