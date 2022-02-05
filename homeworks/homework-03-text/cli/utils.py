from collections import defaultdict
from dataclasses import dataclass, field
import datetime
from pathlib import Path
from time import time_ns
from typing import Sequence
from loguru import logger

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from cli.regressors import regressors


@dataclass
class Params:
    space: Sequence[dict] = field(default_factory=list)
    record: Sequence[str] = field(default_factory=list)


def train_epoch(model, X_train, y_train, X_val, y_val):
    start_time = time_ns()

    logger.info('Started fitting')
    clf = model.fit(X_train, y_train)
    logger.success('Fitting finished')

    train_loss = mean_absolute_error(y_train, clf.predict(X_train))
    val_loss = mean_absolute_error(y_val, clf.predict(X_val))
    logger.info(f'Train MAE: {train_loss}')
    logger.info(f'Validation MAE: {val_loss}')

    elapsed = time_ns() - start_time
    logger.success(f'Finished in {elapsed / 1e9:.3f}s')

    return train_loss, val_loss


@logger.catch
def train(model_name: str, train_path: str, test_path: str, pred_path: str):
    df_train: pd.DataFrame = pd.read_pickle(train_path)
    X_test: pd.DataFrame = pd.read_pickle(test_path)

    logger.success(f'Loaded train data from {train_path} and test data from {test_path}')

    df_train = df_train.sample(frac=1.0)
    X = df_train.drop(columns=['score'])
    y = df_train.score

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1412)

    params: Params = Params()
    if (model_name == 'W2V'):
        corpus = pd.concat((X, X_test))[['positive', 'negative', 'review']]
        params = Params([
            {'corpus': corpus, 'dim': dim}
            for dim in np.linspace(100, 1100, 11, dtype=int)
        ], ['dim'])
    if (model_name == 'FT'):
        corpus = pd.concat((X, X_test))[['positive', 'negative']]
        params = Params([{'corpus': corpus, 'dim': 300}], ['dim'])

    dd = defaultdict(list)
    best_kwargs = {}
    best_loss = 1e9

    if params.space:
        for kwargs in params.space:
            recording = {param: kwargs[param] for param in params.record}
            logger.info(f'Params: {recording}')

            model = regressors[model_name](**kwargs)

            train_loss, val_loss = train_epoch(model, X_train, y_train, X_val, y_val)

            for key in recording:
                dd[key].append(recording[key])
            dd['train_loss'].append(train_loss)
            dd['val_loss'].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_kwargs = kwargs
        else:
            logger.info('Started fitting full dataset')
            model = regressors[model_name](**best_kwargs)
            clf = model.fit(X, y)
            logger.success('Fitting finished')
            X_test['score'] = clf.predict(X_test)
    else:
        model = regressors[model_name]()
        train_loss, val_loss = train_epoch(model, X_train, y_train, X_val, y_val)
        logger.info('Started fitting full dataset')
        clf = model.fit(X, y)
        logger.success('Fitting finished')
        X_test['score'] = clf.predict(X_test)

    save_path = Path(
        pred_path) / f'{datetime.datetime.utcnow().strftime("%Y-%m-%d %H-%M-%S")} {model_name} {train_loss:.3f}-{val_loss:.3f}.csv'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    X_test[['review_id', 'score']].to_csv(save_path, index=False)

    logger.info(f'Saved predictions at {save_path}')
    data = pd.DataFrame.from_dict(dd, orient='columns')

    if len(params.record) > 0:
        plt.figure(figsize=(8, 5 * len(params.record)))
        for i, param in enumerate(params.record):
            plt.subplot(len(params.record), 1, i + 1)
            plt.xlabel(param)
            plt.ylabel('Loss')
            plt.plot(data[param], data['train_loss'], label='Train loss')
            plt.plot(data[param], data['val_loss'], label='Test loss')
        plot_path = Path(
            'plots') / f'{datetime.datetime.utcnow().strftime("%Y-%m-%d %H-%M-%S")} {model_name}.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
