import torch

class NumberGenerator:
    def __init__(self, n_numbers, n_lists, min_value=0, max_value=1, seed=1):
        self.n_numbers = n_numbers
        self.min_value = min_value
        self.max_value = max_value
        torch.manual_seed(seed)
        self.X = torch.zeros(n_lists, n_numbers).uniform_(min_value, max_value)
        ordered_X, permutation = self.X.sort(1)
        self.ordered_X = ordered_X
        self.permutation = permutation

    def __getitem__(self, idx):
        X = self.X[idx]
        ordered_X = self.ordered_X[idx]
        permutation = self.permutation[idx]
        return X, ordered_X, permutation

    def __len__(self):
        return len(self.X)
