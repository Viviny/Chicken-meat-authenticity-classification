from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
import numpy as np


class Classifier(BaseEstimator):
    def __init__(self):
        self.model = BaggingClassifier(bootstrap=False, max_features=0.6)

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict_proba(self, X):
        res = self.model.predict_proba(X)
        return res