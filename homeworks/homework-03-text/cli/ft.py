from collections import Counter, defaultdict
from typing import Sequence
import numpy as np
import pandas as pd
from gensim.models import FastText
from gensim.models.keyedvectors import BaseKeyedVectors
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from cli.regressors import regressors


class FTHelper:
    wv: BaseKeyedVectors
    idf: dict[str, float]
    dim: int
    name: str

    def __init__(self, corpus: Sequence[Sequence[str]], name: str, dim=300):
        self.name = name
        self.dim = dim

        model = FastText(sentences=corpus, size=dim, iter=10, workers=16)
        logger.info(f'Created FastText model')

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

        logger.success(f'Initialized FTHelper for column {name}')


class FT(BaseEstimator):
    def __init__(self, helper: FTHelper):
        self.helper = helper
        self.wv = helper.wv
        self.dim = helper.dim
        self.idf = helper.idf

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ret = X.copy()
        ret = ret.apply(self._process).to_numpy()
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


class FTRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, corpus=pd.DataFrame, dim=300):
        logger.info('Initializing FT')

        logger.info('Initialized pandarallel')

        wvs: dict[str, FTHelper] = {
            column: FTHelper(corpus[column], column, dim) for column in corpus.columns
        }
        logger.info('Initialized wvs dict')

        self.clf = Pipeline([
            ('FT', ColumnTransformer([
                (name, FT(wv), name) for name, wv in wvs.items()
            ])),
            ('regression', Ridge())
        ])
        logger.info('Created a pipeline')

    def fit(self, X, y):
        self.clf = self.clf.fit(X, y)
        return self

    def predict(self, X):
        return np.around(self.clf.predict(X), 1)


regressors['FT'] = FTRegressor
