from models.base_model import BaseModel
import pandas as pd


class Evaluator :
    def __init__(
            self,
            model : BaseModel,
            X_train : pd.DataFrame,
            y_train: pd.DataFrame,
            X_test : pd.DataFrame,
            y_test : pd.DataFrame,
            data_name=None,
    ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.data_name = data_name

    def evaluate(self):
        self.model.fit(self.X_train, self.y_train)
        preds = self.model.predict(self.X_test)
        scores = self.get_scores(preds, self.y_test)
        self._print_scores(scores)
        return scores

    @staticmethod
    def _auc(y_pred, y_true):
        return None # todo

    @staticmethod
    def _aupr(y_pred, y_true):
        return None # todo

    def _get_scores(self, y_pred, y_true):
        scores = {}
        functions = [self._auc, self._aupr]
        names = ['AUC', 'AUPR']
        for name, func in zip(names, functions) :
            scores[name] = func(y_pred, y_true)
        return scores

    def _print_scores(self, scores):
        title = f' Scores for model {self.model.name} '
        if self.data_name is not None:
            title += f'on dataset {self.data_name} '
        bar_len = max(5, int(80 - len(title) / 2))
        print('=' * bar_len + title + '=' * bar_len)
        for score_name in scores.keys():
            print(f'{}:\t{scores[score_name]}')
        print('=' * (len(title) + 2 * bar_len))
