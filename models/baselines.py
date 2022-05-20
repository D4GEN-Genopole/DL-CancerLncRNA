from models.base_model import BaseModel
import numpy as np
import pandas as pd


class RandomModel(BaseModel):
    """Model that predicts randomly 0 or 1 for each class."""
    def __init__(self):
        super().__init__()
        self.n_class = None
        self.target_cols = None

    def fit(self, X, y):
        self.n_class = y.shape[-1]
        self.target_cols = y.columns
        return self

    def predict_proba(self, X):
        N = X.shape[0]
        preds = np.random.randint(2, size=(N, self.n_class))
        return pd.DataFrame(preds, index=X.index, columns=self.target_cols)

    def predict(self, X):
        return self.predict_proba(X)


class LabelMean(BaseModel):
    """Model that predicts the label mean value for each label"""
    def __init__(self):
        super().__init__()
        self.means = None
        self.target_cols = None

    def fit(self, X, y):
        self.means = np.array(y.mean())
        self.target_cols = y.columns
        return self

    def predict_proba(self, X):
        preds = np.vstack([self.means for _ in range(X.shape[0])])
        return pd.DataFrame(preds, index=X.index, columns=self.target_cols)

    def predict(self, X):
        preds = self.predict_proba(X)
        return 1. * (preds >= 0.5)
