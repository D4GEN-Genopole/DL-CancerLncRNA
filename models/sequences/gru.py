from models.base_model import BaseModel
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from models.pytorch.dataset.dataloader import RNADataloader
from models.pytorch.dataset.dataset import RNADataset
from models.pytorch.model.pytorch_model import PytorchModel
from preprocessing.sequences import OneHotEncode, KmersEncoding
import os
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class GRUModel(BaseModel):
    """Model that uses GRU"""
    def __init__(self):
        super(GRUModel, self).__init__()
        self.py_model = None
        self.preprocess = None
        self.chanel = 4

    def fit(self, X, y):
        N = int(0.8*len(X))
        self.columns = y.columns
        X_train, X_val = X.iloc[:N], X.iloc[N:]
        y_train, y_val = y.iloc[:N], y.iloc[N:]
        self.preprocess = OneHotEncode()
        # self.preprocess = KmersEncoding(4)
        self.preprocess.fit(X_train)
        X_train = self.preprocess.transform(X_train)
        # print(f"SIZE X_TRAIN : {X_train.shape}")
        X_val = self.preprocess.transform(X_val)
        dataset_train = RNADataset(X_train,y_train)
        dataset_val = RNADataset(X_val, y_val)
        self.params_dataloader = {
            "device": DEVICE,
            "batch_size": 4,
            "shuffle": True,
            "dataset_train": dataset_train,
            "dataset_val": dataset_val,
            "dataset_test": dataset_val,
        }
        self.dataloader = RNADataloader(**self.params_dataloader)
        model = GRUModule(self.chanel, 128, 35)
        # model = LSTMModule(self.chanel, 256, 35)
        hp_pl = {
            'lr' : 1e-4,
            'model' : model,
        }
        self.py_model = PytorchModel(**hp_pl)
        params_trainer = {
                "max_epochs": 10,
            }
        if 'cpu' not in DEVICE.type:
            params_trainer['gpus'] = -1
        trainer = Trainer(**params_trainer)
        with wandb.init(project='d4gen', entity='sayby', config=hp_pl):
            trainer.fit(self.py_model, self.dataloader)
        self.py_model.model.save_checkpoint()

    def predict(self, X):
        if self.py_model is not None:
            self.py_model.model.load_checkpoint()
        else:
            print('NOT LOADED WEIGHTS')
            return None
        if self.preprocess is None :
            print("NOT INITIALISE PREPROCESS")
        else:
            X_test = self.preprocess.transform(X)
            self.params_dataloader['dataset_test'] = RNADataset(X_test, None)
            self.params_dataloader['batch_size'] = 1
            self.dataloader = RNADataloader(**self.params_dataloader)
            outputs = []
            for batch in self.dataloader.test_dataloader():
                x,_ = batch
                x = x.to(DEVICE)
                output = self.py_model.model.to(DEVICE)(x)
                output = list(output.detach().cpu().numpy()[0])
                outputs.append(output)
            preds = pd.DataFrame(outputs, index = X.index, columns = self.columns)
            preds.to_csv('GRU.csv')
            return preds

    def predict_proba(self, X):
        return self.predict(X)


class GRUModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUModule, self).__init__()

        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=2,
                          batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.checkpoint = os.path.join('weights', 'gru','gru.pth')

    def forward(self, x, gpu=True):
        print(f"FORWARD : {x.shape}")
        gru_output, h_n = self.gru(x.float())
        out = self.out(gru_output)[:, -1, :]
        print(f"FORWARD OUTPUT : {out.shape}")
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

class LSTMModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModule, self).__init__()
        self.input_dim = input_dim
        self.norm = self.norm = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1,
                            batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.checkpoint = os.path.join('weights', 'model', 'lstm.pth')

    def forward(self, x, gpu=True):
        # print(f"FORWARD : {x.shape}")
        x = self.norm(x.float())
        x = x.view(-1, 4, self.input_dim)
        lstm_output, _ = self.lstm(x)
        out = self.out(lstm_output)[:, -1, :]
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

