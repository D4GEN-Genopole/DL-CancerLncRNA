import os
import numpy as np
import pandas as pd
from models.base_model import BaseModel
from sklearn.metrics import roc_auc_score


class Evaluator :
    def __init__(
            self,
            model : BaseModel,
            X_train : pd.DataFrame,
            y_train: pd.DataFrame,
            X_test : pd.DataFrame,
            y_test : pd.DataFrame,
            data_name=None,
            path=None,
    ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.data_name = data_name
        self.path = path
        if path is not None:
            self.model.load(path)


    def evaluate(self):
        if self.path is None:
            self.model.fit(self.X_train, self.y_train)
        preds = self.model.predict(self.X_test)
        scores = self._get_scores(preds, self.y_test)
        self._print_scores(scores)
        return scores

    @staticmethod
    def _auc(y_pred, y_true):
        auc = roc_auc_score(np.array(y_true), np.array(y_pred))
        return auc

    @staticmethod
    def _aupr(y_pred, y_true):
        return 0. # todo

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
            print(f'{score_name}:\t{scores[score_name]}')
        print('=' * (len(title) + 2 * bar_len))


class EvaluatorFromPaths(Evaluator):
    def __init__(self, model, train_path, test_path, n_targets, index=None, **kwargs):
        df_train, df_test = pd.read_csv(train_path), pd.read_csv(test_path)
        if index is not None:
            df_train.set_index(index, inplace=True)
            df_test.set_index(index, inplace=True)
        X_train, y_train = df_train.iloc[:, :-n_targets], df_train.iloc[:, -n_targets:]
        X_test, y_test = df_test.iloc[:, :-n_targets], df_test.iloc[:, -n_targets:]
        super().__init__(model, X_train, y_train, X_test, y_test, **kwargs)


class SequencesEvaluator(EvaluatorFromPaths):
    def __init__(self, model, **kwargs):
        super().__init__(model,
                         os.path.join('data', 'sequences_train.csv'),
                         os.path.join('data', 'sequences_test.csv'),
                         n_targets=35,
                         index='gencode_id',
                         data_name='sequences',
                         **kwargs)


class ExpressionsEvaluator(EvaluatorFromPaths):
    def __init__(self, model, **kwargs):
        super().__init__(model,
                         os.path.join('data', 'expressions_train.csv'),
                         os.path.join('data', 'expressions_test.csv'),
                         n_targets=35,
                         index='gencode_id',
                         data_name='expressions',
                         **kwargs)


class SequencesExpressionsEvaluator(EvaluatorFromPaths):
    def __init__(self, model, **kwargs):
        super().__init__(model,
                         os.path.join('data', 'sequences_expressions_train.csv'),
                         os.path.join('data', 'sequences_expressions_test.csv'),
                         n_targets=35,
                         index='gencode_id',
                         data_name='sequences and expressions',
                         **kwargs)
