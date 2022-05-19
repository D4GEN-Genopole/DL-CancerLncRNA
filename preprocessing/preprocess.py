class Preprocess:
    def __init__(self):
        pass
    
    def transform(self, X) :
        raise NotImplementedError

    def fit(self, X):
        raise NotImplementedError

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
