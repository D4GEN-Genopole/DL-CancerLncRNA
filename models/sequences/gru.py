from models.base_model import BaseModel
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from models.pytorch.dataset.dataloader import RNADataloader
from models.pytorch.dataset.dataset import RNADataset
from models.pytorch.model.pytorch_model import PytorchModel
from preprocessing.sequences import OneHotEncode
import os
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class GRUModel(BaseModel):
    """Model that uses GRU"""
    def __init__(self):
        super(GRUModel, self).__init__()
        self.py_model = None
        self.preprocess = None

    def fit(self, X, y):
        N = int(0.8*len(X))
        X_train, X_val = X.iloc[:N], X.iloc[N:]
        y_train, y_val = y.iloc[:N], y.iloc[N:]
        self.preprocess = OneHotEncode()
        self.preprocess.fit(X_train)
        X_train = self.preprocess.transform(X_train)
        X_val = self.preprocess.transform(X_val)
        dataset_train = RNADataset(X_train,y_train)
        dataset_val = RNADataset(X_val, y_val)
        params_dataloader = {
            "device": DEVICE,
            "batch_size": 4,
            "shuffle": True,
            "dataset_train": dataset_train,
            "dataset_val": dataset_val,
            "dataset_test": dataset_val,
        }
        self.dataloader = RNADataloader(**params_dataloader)
        gru_module = GRUModule(4, 128, 35)
        hp_pl = {
            'lr' : 1e-3,
            'model' : gru_module,

        }
        self.py_model = PytorchModel(**hp_pl)
        params_trainer = {
                "max_epochs": 20,
            }
        if 'cpu' not in DEVICE.type :
            params_trainer['gpus'] = -1
        trainer = Trainer(**params_trainer)
        with wandb.init(project='d4gen', entity='sayby', config=hp_pl):
            trainer.fit(self.py_model, self.dataloader)
        self.py_model.model.save_checkpoint()

    def predict(self, X):
        if self.py_model is not None :
            self.py_model.model.load_checkpoint()
        else :
            print('NOT LOADED WEIGHTS')
            return None
        if self.preprocess is None :
            print("NOT INITIALISE PREPROCESS")
        else:
            X_test = self.preprocess.transform(X)
            dataset = RNADataset(X_test, None)
            outputs = []
            for batch in dataset:
                x,_ = batch
                x = x.to(DEVICE).reshape(-1,300 , 4)
                output = self.py_model.model.to(DEVICE)(x)
                outputs.append(list(output.detach().cpu().numpy()[0]))
            return pd.DataFrame(outputs)

    def predict_proba(self, X):
        return self.predict(X)


class GRUModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUModule, self).__init__()

        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1,
                          batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.checkpoint = os.path.join('weights', 'gru','gru.pth')

    def forward(self, x, gpu=True):
        gru_output, h_n = self.gru(x.float())
        out = self.out(gru_output)[:, -1, :]
        return out

    def save_checkpoint(self):
        print('--- Save model checkpoint ---')
        torch.save(self.state_dict(), self.checkpoint)

    def load_checkpoint(self, gpu=True):
        print('--- Loading model checkpoint ---')
        if torch.cuda.is_available() and gpu:
            self.load_state_dict(torch.load(self.checkpoint))
        else:
            self.load_state_dict(torch.load(self.checkpoint, map_location=torch.device('cpu')))
