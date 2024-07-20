import torch
import config

def plain_rms_norm(x):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True))
    return x / (rms + config.normalization_eps)


if __name__ == "__main__":
    import math
    print("config.normalization_eps:", config.normalization_eps)
    
    a = torch.randn(config.batch_size, config.context_window, config.emb_size, dtype=torch.float64, device="cuda")
    print("a.shape:", a.shape)
    
    out_1 = plain_rms_norm(a)
    
    norm_a = a.norm(dim=-1, keepdim=True)
    out_2 = math.sqrt(config.emb_size) * a / norm_a
    
    print("when config.normalization_eps is zero, expect close to 0.0:")
    print((out_2 - out_1).abs().max().item())
    

# [x] proofreading
# [x] testing
