class BaseModel:
    def __init__(self, name=None):
        self.name = name if name is not None else self.__class__.__name__

    def fit(self, X, y):
        return self

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError
