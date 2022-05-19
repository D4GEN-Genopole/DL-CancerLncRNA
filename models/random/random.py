from models.base_model import BaseModel
import numpy as np
import pandas as pd

class RandomModel(BaseModel):
    """Model that predicts randomly 0 or 1 for each class."""
    def __init__(self, name="Random"):
        super(RandomModel, self).__init__()

    def fit(self, X, y):
        self.n_class = y.shape[-1]
        self.target_cols = y.columns
        pass

    def predict(self, X):
        N = X.shape[0]
        pred = np.random.random_integers(1, size=(N,self.n_class))
        return pd.DataFrame(pred, index=X.index, columns=self.target_cols)

    def predict_proba(self, X):
        return self.predict(X)
