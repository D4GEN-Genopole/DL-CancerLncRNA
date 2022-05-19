from models.base_model import BaseModel
import pandas as pd

class Evaluate :
    def __init__(
            self,
            model : BaseModel,
             X_train : pd.DataFrame,
             y_train: pd.DataFrame,
             X_test : pd.DataFrame,
             y_test : pd.DataFrame,
    ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        return self.model.predict(self.X_test)

    def auc(self, predict, target):
        return None

    def aupr(self, predict, target):
        return None

    def score(self, predict, target):
        scores = {}
        functions = [self.auc, self.aupr]
        names = ['AUC', 'AUPR']
        for name, func in zip(names, functions) :
            scores[name] = func(predict, target)
        return scores



