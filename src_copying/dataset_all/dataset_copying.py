import torch

# note: the number of different characters* in a string is 26 (alphabet)
#     *: excluding three special tokens: <BOS>, <SEP>, and <EOS>
class DatasetCopying(torch.utils.data.Dataset):
    def __init__(self, batch_size, string_size):
        self.batch_size = batch_size
        self.string_size = string_size

    def __len__(self):
        return 1000000000000000

    def __getitem__(self, idx_batch):
        # construct string_all
        string_all = torch.randint(low=3, high=29, size=(self.batch_size, self.string_size))  # 29 = 26 + 3
        # initialize X and Y
        X = torch.zeros((self.batch_size, 2 * (self.string_size + 1)), dtype=torch.int64)
        Y = torch.zeros((self.batch_size, 2 * (self.string_size + 1)), dtype=torch.int64)
        # insert string_all into X and Y
        X[:, 1:self.string_size + 1] = string_all
        X[:, self.string_size + 2:] = string_all
        Y[:, self.string_size + 1:-1] = string_all
        # insert special tokens into X and Y
        X[:, 0] = 0
        X[:, self.string_size + 1] = 1
        Y[:, :self.string_size + 1] = -1
        Y[:, -1] = 2
        return X, Y


if __name__ == "__main__":
    import config
    print("config.batch_size:", config.batch_size)
    print("config.string_size:", config.string_size)
    dataset = DatasetCopying(config.batch_size, config.string_size)
    x, y = dataset[0]
    print("x.shape:", x.shape)
    print("y.shape:", y.shape)

# checked
