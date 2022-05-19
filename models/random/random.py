from models.base_model import BaseModel
import numpy as np

class RandomModel(BaseModel):
    """Model that predicts randomly 0 or 1 for each class."""
    def __init__(self, name="Random"):
        super(RandomModel, self).__init__()

    def fit(self, X, y):
        self.n_class = y.shape[-1]
        pass

    def predict(self, X):
        N = X.shape[0]
        pred = np.random.random_integers(1, size=(N,self.n_class))
        return pred

    def predict_proba(self, X):
        return self.predict(X)
