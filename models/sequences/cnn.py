import pandas as pd
from models.base_model import BaseModel
from preprocessing.sequences import MersIndexEncoding
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense


class EmbeddingCNN1D(BaseModel):
    """Model based on a keras CNN1D with an Embedding layer"""
    def __init__(self, k=1, embedding_size=100, length=500, n_filters=32,
                       kernel_size=5, dense_size=10, epochs=2):
        super().__init__()
        self.k = k
        self.embedding_size = embedding_size
        self.length = length
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dense_size = dense_size
        self.epochs = epochs
        self.target_cols = None
        self.model = None
        self.preprocessor = MersIndexEncoding(self.k, length=self.length)

    def fit(self, X, y):
        self.target_cols = y.columns
        vocab_size = 4**self.k + 1
        model = Sequential()
        model.add(Embedding(vocab_size, self.embedding_size, input_length=self.length))
        model.add(Conv1D(self.n_filters, self.kernel_size, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(self.dense_size, activation='relu'))
        model.add(Dense(len(self.target_cols), activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy')
        self.model = model

        #################### this block to be removed
        import pandas as pd
        df_test = pd.read_csv('data\sequences_test.csv')
        df_test.set_index('gencode_id', inplace=True)
        X_test, y_test = df_test.iloc[:, :-35], df_test.iloc[:, -35:]
        X_test_preprocessed = self.preprocessor.transform(X_test)
        ####################until here

        X_preprocessed = self.preprocessor.transform(X)
        history = self.model.fit(X_preprocessed.values, y.values,
                                 epochs=self.epochs,
                                 validation_data=(X_test_preprocessed.values, y_test.values), ###################### to remove
                                 batch_size=32)
        return self

    def predict_proba(self, X):
        X_preprocessed = self.preprocessor.transform(X)
        preds = self.model.predict(X_preprocessed.values)
        return pd.DataFrame(preds, index=X.index, columns=self.target_cols)