import datetime
from pathlib import Path
from loguru import logger

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from cli.regressors import regressors


@logger.catch
def train(model_name: str, train_path: str, test_path: str, pred_path: str):
    model = regressors[model_name]()
    save_path = Path(pred_path) / f'{model_name} {datetime.datetime.utcnow().strftime("%Y-%m-%d %H-%M-%S")}.csv'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df_train: pd.DataFrame = pd.read_pickle(train_path)
    X_test: pd.DataFrame = pd.read_pickle(test_path)

    logger.success(f'Loaded train data from {train_path} and test data from {test_path}')

    X = df_train.drop(columns=['score'])
    y = df_train.score

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1412)

    logger.info('Fitting train data')

    clf = model.fit(X_train, y_train)

    logger.success('Fitting finished')

    logger.info(f'Train MAE: {mean_absolute_error(y_train, clf.predict(X_train))}')
    logger.info(f'Test MAE: {mean_absolute_error(y_val, clf.predict(X_val))}')

    X_test['score'] = clf.predict(X_test)
    X_test[['review_id', 'score']].to_csv(save_path, index=False)

    logger.success(f'Saved predictions at {save_path}')
