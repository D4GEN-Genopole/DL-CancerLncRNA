class BasePreprocessor:
    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X) :
        raise NotImplementedError

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
