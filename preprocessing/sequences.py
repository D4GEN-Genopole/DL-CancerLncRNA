from preprocessing.base_preprocessor import BasePreprocessor
import pandas as pd

NUCLEOTIDE_DICT = {
            "A" : [1,0,0,0],
            "C" : [0,1,0,0],
            "G" : [0,0,1,0],
            "T" : [0,0,0,1]
        }

class OneHotEncode(BasePreprocessor):
    def __init__(self):
        super(OneHotEncode, self).__init__()

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

