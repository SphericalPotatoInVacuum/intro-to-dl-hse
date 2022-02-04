import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
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
            ('regression', Ridge())
        ])

    def fit(self, X, y):
        self.clf = self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)


regressors['TFIDF'] = TfIdfRegressor
