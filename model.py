import pandas as pd
from sklearn.svm import svc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

class model:
    def __init__(self):
        self.model = None
        print("Hello world")

    def train(self, X, y):
        self.model = svc()
        self.model.fit(X, y)

    def predict(self, X):
        if self.model is not None:
            return self.model.predict(X)
        else:
            raise Exception("Model has not been trained yet.")

    def evaluate(self, X, y):
        if self.model is not None:
            predictions = self.predict(X)
            return accuracy_score(y, predictions)
        else:
            raise Exception("Model has not been trained yet.")


