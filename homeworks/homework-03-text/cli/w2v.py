from collections import Counter, defaultdict
from typing import Sequence

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.keyedvectors import BaseKeyedVectors
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from cli.regressors import regressors


class WVHelper:
    wv: BaseKeyedVectors
    idf: dict[str, float]
    dim: int
    name: str

    def __init__(self, corpus: Sequence[Sequence[str]], name: str, dim=300):
        self.name = name
        self.dim = dim

        model = Word2Vec(sentences=corpus, size=dim, seed=42, workers=16)
        logger.info(f'Created word2vec model')

        self.wv: BaseKeyedVectors = model.wv
        del model

        cnt: Counter = Counter()
        for words in corpus:
            for word in set(words):
                cnt[word] += 1
        logger.info('Counted words')

        self.idf: defaultdict[str, int] = defaultdict(
            float,
            {word: np.log((len(corpus) / cnt[word])) for words in corpus for word in words}
        )
        logger.info('Calculated idf statistics')

        logger.success(f'Initialized WVHelper for column {name}')


class W2V(BaseEstimator):
    def __init__(self, helper: WVHelper):
        self.helper = helper
        self.wv = helper.wv
        self.dim = helper.dim
        self.idf = helper.idf

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ret = X.copy()
        ret = ret.parallel_apply(self._process).to_numpy()
        ret = np.stack(ret)
        return ret

    def _process(self, words):
        vectors = np.array([self.wv.get_vector(word).round(2) for word in words if word in self.wv])
        if len(vectors) == 0:
            return np.zeros(self.dim)
        weights = np.array([self.idf[word] for word in words if word in self.wv])
        return np.average(vectors, axis=0, weights=weights)

    def get_params(self, deep=True):
        return {'helper': self.helper}


class W2VRegressor(BaseEstimator, RegressorMixin):
    need_preprocessing = True

    def __init__(self, corpus=pd.DataFrame, dim=300):
        logger.info('Initializing W2V')
        from pandarallel import pandarallel
        pandarallel.initialize(use_memory_fs=False, verbose=1)

        wvs: dict[str, WVHelper] = {
            column: WVHelper(corpus[column], column, dim) for column in corpus.columns
        }

        self.clf = Pipeline([
            ('w2v', ColumnTransformer([
                (name, W2V(wv), name) for name, wv in wvs.items()
            ])),
            ('regression', Ridge())
        ])

    def fit(self, X, y):
        self.clf = self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)


regressors['W2V'] = W2VRegressor
