import numpy as np
from torch.utils.data import Dataset
import torch
from preprocessing.sequences import OneHotEncode


class RNADataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        input_ = torch.from_numpy(np.array(self.X[idx]))
        if self.y is not None :
            label_ = torch.from_numpy(self.y.iloc[idx].values).reshape(-1,1)
        else:
            label_ = None
        return input_, label_
