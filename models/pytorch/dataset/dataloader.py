from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torch


class RNADataloader(LightningDataModule):
    def __init__(self,**kwargs):
        super().__init__()
        self.setup_params(**kwargs)
        self.setup_datasets(**kwargs)

    def setup_params(self, device, batch_size, shuffle, **kwargs):
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle

    def setup_datasets(self,dataset_train, dataset_val, dataset_test, **kwargs):
        self.dataset_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle)
        self.dataset_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=self.shuffle)
        self.dataset_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=self.batch_size,
            shuffle=self.shuffle)


    def train_dataloader(self):
        return self.dataset_train

    def val_dataloader(self):
        return self.dataset_val

    def test_dataloader(self):
        return self.dataset_test