import pandas as pd
import itertools
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
    def convert_to_char(X):
        """Convert each sequence to list of characters."""
        return [x for x in X[0]]

    @staticmethod
    def convert_to_one_hot(X_process):
        """Convert to one hot encoding for the sequences."""
        new_train = []
        for i, row in X_process.iterrows():
            new_train.append(
                [NUCLEOTIDE_DICT[x] for x in row[0]]
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
        self.mers = [''.join(comb) for comb in itertools.combinations_with_replacement(
                                                        ['A', 'T', 'C', 'G'], k)]

    def transform(self, X) :
        df_count = pd.DataFrame({mer: X.sequence.apply(lambda x: x.count(mer))
                                                        for mer in self.mers},
                                                        index=X.index)
        df_per_kb = df_count.div(X.sequence.apply(len) / 1000, axis=0)
        df_normalized = (df_per_kb - df_per_kb.mean()) / df_per_kb.std()
        return df_normalized
