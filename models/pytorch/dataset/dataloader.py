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

    def collate_fn(self, batch):
        """Collate fn for the dataset.

        Args :
            -batch : the batch with 4 tensors (input_ids, attention_mask, labels)
        """
        input_ids, attention_mask, target_ids = [], [], []
        for subbatch in batch:
            input_ids.append(subbatch[0])
            attention_mask.append(subbatch[1])
            target_ids.append(subbatch[2])

        input_ids = torch.vstack(input_ids)
        attention_mask = torch.vstack(attention_mask)
        target_ids = torch.vstack(target_ids)

        return input_ids, attention_mask, target_ids


    def collate_fn(self, batch):
        """Collate for the dataset."""
        x, y = [], []
        for subbatch in batch :
            x.append()


    def train_dataloader(self):
        return self.dataset_train

    def val_dataloader(self):
        return self.dataset_val

    def test_dataloader(self):
        return self.dataset_test