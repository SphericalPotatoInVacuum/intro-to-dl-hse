import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from cli.regressors import regressors


class TfIdfRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.clf = Pipeline([
            ('tfidf', ColumnTransformer([
                ('tfidf_p', TfidfVectorizer(preprocessor=lambda x: ' '.join(x)), 'positive'),
                ('tfidf_n', TfidfVectorizer(preprocessor=lambda x: ' '.join(x)), 'negative'),
                ('tfidf', TfidfVectorizer(preprocessor=lambda x: ' '.join(x)), 'review'),
            ])),
            ('regression', LogisticRegression(max_iter=10000))
        ])

    def fit(self, X, y):
        self.clf = self.clf.fit(X, (y * 10).astype(int))
        return self

    def predict(self, X):
        weights = self.clf.predict_proba(X)
        return weights @ self.clf.classes_ / 10


regressors['TFIDF'] = TfIdfRegressor
