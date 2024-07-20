import torch
import config
from dataset_all.dataset_copying import DatasetCopying


# ----- #
# how to calculate accuracy
# ----- #
# 1. <EOS> wrong location: 0%
#        condition_1 = y_sample[:, -1] == <EOS>
#        condition_2 = <EOS> not in y_sample[:, config.string_size + 1:-1]
# 2. <EOS> correct location:
#        accuracy of
#            y_sample[:, config.string_size + 1:-1]
#            vs
#            y[:, config.string_size + 1:-1]
# note: calculate_accuracy_copying can be used for both
#     soft and hard accuracies
def calculate_accuracy_copying(p, y):
    # get accuracy
    string_y = y[:, config.string_size + 1:-1]
    string_p = p[:, config.string_size + 1:-1]
    accuracy_all = (string_y == string_p).float().mean(dim=-1)
    # consider <EOS>
    mask_1 = p[:, -1] != 2
    mask_2 = (p[:, config.string_size + 1:-1] == 2).any(dim=-1)
    mask = torch.logical_or(mask_1, mask_2)
    accuracy_all[mask] = 0.0
    return accuracy_all

if __name__ == "__main__":
    print("config.batch_size:", config.batch_size)
    print("config.string_size:", config.string_size)
    # get dataloader
    dataset = DatasetCopying(config.batch_size * config.num_grad_accu, config.string_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=config.num_worker_dataloader, pin_memory=True)
    # get y and p
    for x, y in dataloader:
        break
    p = y.clone()
    print("y.shape:", y.shape)
    print("p.shape:", p.shape)
    
    print("Expect all ones:")
    print((calculate_accuracy_copying(p, y) == 1).all())
    
    print("Expect all zeros:")
    p[:, -1] = 1
    print((calculate_accuracy_copying(p, y) == 0).all())
    
    # other
    p = y.clone()
    p[0, 40] = 2
    print(calculate_accuracy_copying(p, y))
    
    p = y.clone()
    p[0, 40] = 10
    y[0, 40] = 11
    print(calculate_accuracy_copying(p, y))

# checked
