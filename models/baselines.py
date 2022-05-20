import numpy as np
import pandas as pd
from models.base_model import BaseModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


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


class SklearnModel(BaseModel):
    """Model based on a sklearn classifier"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.preprocessor = None
        self.target_cols = None

    def fit(self, X, y):
        if self.preprocessor is not None:
            X = self.preprocessor.fit_transform(X)
        self.model.fit(X, y)
        self.target_cols = y.columns
        return self

    def predict_proba(self, X):
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)
        preds = self.model.predict(X)
        return pd.DataFrame(preds, index=X.index, columns=self.target_cols)


class RF(SklearnModel):
    """Random forest classifier"""
    def __init__(self):
        super().__init__(RandomForestClassifier())


class KNN(SklearnModel):
    """K-nearest neighbors classifier"""
    def __init__(self):
        super().__init__(KNeighborsClassifier())


class MLP(SklearnModel):
    """Multi-layer perceptron classifier"""
    def __init__(self):
        super().__init__(MLPClassifier())
