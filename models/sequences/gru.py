from models.base_model import BaseModel
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.pytorch.dataset.dataloader import RNADataloader
from models.pytorch.dataset.dataset import RNADataset
from preprocessing.sequences import OneHotEncode

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class GRU(BaseModel):
    """Model that uses GRU"""
    def __init__(self):
        super(GRU, self).__init__()

    def fit(self, X, y):
        N = int(0.8*len(X))
        X_train, X_val = X.iloc[:N], X.iloc[N:]
        y_train, y_val = y.iloc[:N], y.iloc[N:]
        preprocess = OneHotEncode().fit_transform(X_train)
        dataset_train = RNADataset(preprocess,y_train)
        dataset_val = RNADataset(X_val, y_val)
        params_dataloader = {
            "device": DEVICE,
            "batch_size": 1,
            "shuffle": True,
            "dataset_train": dataset_train,
            "dataset_val": dataset_val,
            "dataset_test": dataset_val,
        }
        dataloader = RNADataloader(**params_dataloader)
        train = dataloader.train_dataloader()
        for batch in train :
            print(batch)
            break


    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass
