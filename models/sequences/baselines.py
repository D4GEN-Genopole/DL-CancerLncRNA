from preprocessing.sequences import KmersEncoding
from models.baselines import RandomModel, LabelMean, RF, KNN, MLP


class SeqRandomModel(RandomModel):
    def __init__(self):
        super().__init__()
        self.preprocessor = KmersEncoding(3)


class SeqLabelMean(LabelMean):
    def __init__(self):
        super().__init__()
        self.preprocessor = KmersEncoding(3)


class KmersRF(RF):
    def __init__(self):
        super().__init__()
        self.preprocessor = KmersEncoding(3)


class KmersKNN(KNN):
    def __init__(self):
        super().__init__()
        self.preprocessor = KmersEncoding(3)


class KmersMLP(MLP):
    def __init__(self):
        super().__init__()
        self.preprocessor = KmersEncoding(3)
