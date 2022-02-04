from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from vecto.embeddings import load_from_dir
import gensim.downloader
from scipy.special import softmax
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from cli.regressors import regressors


class W2V(BaseEstimator):
    def __init__(self, vsm):
        self.vsm = vsm

    def fit(self, X, y=None):
        self.mean = y.mean()
        self.cnt = Counter()
        self.idf = defaultdict(int)
        for words in X:
            for word in words:
                self.cnt[word] += 1
        self.idf.update({word: np.log((len(X) / self.cnt[word]) if self.cnt[word] > 0 else 1)
                         for words in X for word in words})
        return self

    def transform(self, X):
        ret = X.copy()
        ret = ret.apply(self._process).to_numpy()
        arr = np.array([x.shape for x in ret])
        assert (arr[0] == arr).all()
        ret = np.stack(ret)
        return ret

    def _process(self, words):
        if len(words) == 0:
            return np.zeros(500)
        vectors = np.array([self.vsm.get_vector(word) for word in words])
        weights = softmax(np.array([self.idf[word] for word in words])).reshape((-1, 1))
        return (vectors.T @ weights).T.flatten()


class W2VRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        from pandarallel import pandarallel
        pandarallel.initialize()
        model_path = 'word_linear_glove_500d'
        vsm = load_from_dir(model_path)
        vsm.normalize()
        logger.success(f'Loaded vsm from {model_path}')

        self.clf = Pipeline([
            ('w2v', ColumnTransformer([
                ('w2v', W2V(vsm), 'review'),
            ])),
            ('regression', Ridge())
        ])

    def fit(self, X, y):
        self.clf = self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)


regressors['W2V'] = W2VRegressor
