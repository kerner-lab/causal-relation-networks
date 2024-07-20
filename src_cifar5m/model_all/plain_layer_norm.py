import torch
import config


def plain_layer_norm(x):
    std, mean = torch.std_mean(x, unbiased=False, dim=-1, keepdim=True)
    return (x - mean) / (std + config.normalization_eps)

if __name__ == "__main__":
    print("config.normalization_eps:", config.normalization_eps)
    
    a = torch.randn(config.batch_size, config.context_window, config.emb_size, dtype=torch.float64, device="cuda")
    print("a.shape:", a.shape)

    out_1 = plain_layer_norm(a)
    
    mean = a.mean(dim=-1, keepdim=True)
    std = a.std(dim=-1, keepdim=True, unbiased=False)
    out_2 = (a - mean) / (std + config.normalization_eps)

    print("expect close to 0.0:", (out_2 - out_1).abs().max().item())

    
# [x] proofreading
# [x] testing
