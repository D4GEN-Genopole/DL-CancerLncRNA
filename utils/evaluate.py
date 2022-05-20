import os
import numpy as np
import pandas as pd
from models.base_model import BaseModel
from sklearn.metrics import roc_auc_score, f1_score, coverage_error, \
                            label_ranking_average_precision_score, label_ranking_loss


class Evaluator :
    def __init__(
            self,
            model : BaseModel,
            X_train : pd.DataFrame,
            y_train : pd.DataFrame,
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
        preds = self.model.predict_proba(self.X_test)
        target_sets = ['cancer', 'functions', 'all targets']
        target_scores = []
        for target in target_sets:
            scores = self._get_scores(preds, self.y_test, target=target)
            target_scores.append(scores)
        self._print_scores(target_scores, target_sets)
        return scores

    @staticmethod
    def _auc(y_pred, y_true):
        auc = roc_auc_score(y_true, y_pred, average='micro')
        return auc

    @staticmethod
    def _f1_score(y_pred, y_true):
        return f1_score(y_true, y_pred, average='macro')

    @staticmethod
    def _lrap(y_pred, y_true):
        return label_ranking_average_precision_score(y_true, y_pred)

    @staticmethod
    def _neg_coverage_error(y_pred, y_true):
        return 1 - coverage_error(y_true, y_pred) / y_true.shape[1]

    @staticmethod
    def _neg_ranking_loss(y_pred, y_true):
        return 1 - label_ranking_loss(y_true, y_pred)

    def _get_scores(self, y_pred, y_true, target='all targets'):
        if target == 'cancer':
            y_pred, y_true = y_pred.iloc[:, :-5], y_true.iloc[:, :-5]
        elif target == 'functions':
            y_pred, y_true = y_pred.iloc[:, -5:], y_true.iloc[:, -5:]
        scores = {}
        functions = [
                     # self._auc, self._f1_score,
                     self._lrap, self._neg_coverage_error, self._neg_ranking_loss
                     ]
        names = [
                 # 'AUC', 'F1',
                 'LRAP', 'Negative coverage error', 'Negative ranking loss'
                 ]
        for name, func in zip(names, functions):
            scores[name] = func(y_pred, y_true)
        return scores

    def _print_scores(self, target_scores, target_sets):
        title = f' Scores for model {self.model.name} '
        if self.data_name is not None:
            title += f'on {self.data_name} dataset '
        bar_len = max(5, int(40 - len(title) / 2))
        print('=' * bar_len + title + '=' * bar_len)
        for i, (scores, target) in enumerate(zip(target_scores, target_sets)):
            lineskip = '\n' if i > 0 else ''
            print(f'{lineskip}When predicting {target}:')
            for score_name in scores.keys():
                space_len = max(1, 25 - len(score_name))
                print(f'{score_name}:{" " * space_len}{scores[score_name]:.2f}')
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
