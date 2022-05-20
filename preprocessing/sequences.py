import numpy as np
import pandas as pd
import itertools, more_itertools
from preprocessing.base_preprocessor import BasePreprocessor

NUCLEOTIDE_DICT = {
            "A" : [1,0,0,0],
            "C" : [0,1,0,0],
            "G" : [0,0,1,0],
            "T" : [0,0,0,1]
        }

class OneHotEncode(BasePreprocessor):
    def __init__(self):
        super().__init__()

    @staticmethod
    def add_pad(x, N):
        """Make sequence of N size. Add 0 if necessary."""
        if len(x) > N:
            return x[:N]
        else :
            pad = N-len(x)
            return x + [[0,0,0,0] for i in range(pad)]

    @staticmethod
    def convert_to_char(X):
        """Convert each sequence to list of characters."""
        return [x for x in X[0]]

    @staticmethod
    def convert_to_one_hot(X_process, N=1000):
        """Convert to one hot encoding for the sequences."""
        new_train = []
        for i, row in X_process.iterrows():
            n_train = [NUCLEOTIDE_DICT[x] for x in row[0]]
            n_train = OneHotEncode.add_pad(n_train, N)
            new_train.append(
                n_train
            )
        return new_train

    def fit(self, X):
        pass

    def transform(self, X):
        X_process = pd.DataFrame(X.apply(self.convert_to_char, axis=1))
        X_one_hot = self.convert_to_one_hot(X_process)
        return X_one_hot


    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class KmersEncoding(BasePreprocessor):
    def __init__(self, k):
        super().__init__()
        self.mers = [''.join(comb) for comb in itertools.product('ATCG', repeat=k)]

    def transform(self, X):
        df_count = pd.DataFrame({mer: X.sequence.apply(lambda x: x.count(mer))
                                                        for mer in self.mers},
                                                        index=X.index)
        df_per_kb = df_count.div(X.sequence.apply(len) / 1000, axis=0)
        df_normalized = (df_per_kb - df_per_kb.mean()) / df_per_kb.std()
        return df_normalized


class MersOneHotEncoding(BasePreprocessor):
    def __init__(self, k, length=500):
        super().__init__()
        self.k = k
        self.length = length - 2
        mers = [''.join(comb) for comb in itertools.product('ATCG', repeat=self.k)]
        self.val_dict = {mer: i+1 for i, mer in enumerate(mers)}
        self.val_dict['start'] = 0
        self.val_dict['end'] = max(self.val_dict.values()) + 1

    def transform(self, X):
        X.loc[:, 'sequence'] = X.sequence.apply(lambda x: x[:self.length + self.k - 1])
        X = X.sequence.apply(lambda x: ['start'] + [''.join(comb)
                                    for comb in more_itertools.windowed(x, self.k)]
                                    + ['end'] + [np.nan] * (self.length + self.k - 1 - len(x))
                                    if len(x) >= self.k
                                    else ['start', 'end'] + [np.nan] * self.length)
        X = pd.DataFrame(X.tolist())
        X = X.applymap(lambda x: self.val_dict[x] if not pd.isna(x) else x)
        nan_mask = X.notna()
        X = pd.DataFrame(X.fillna(0), dtype='int')
        xmat = np.stack((np.arange(X.shape[0]),) * X.shape[1], axis=1)
        ymat = np.stack((np.arange(X.shape[1]),) * X.shape[0], axis=0)
        values = np.zeros((X.shape[0], len(self.val_dict.keys()), self.length+2))
        values[xmat[nan_mask], X.values[nan_mask], ymat[nan_mask]] = 1.

        return values
