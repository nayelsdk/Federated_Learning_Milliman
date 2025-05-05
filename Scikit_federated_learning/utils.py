import numpy as np
import pandas as pd
from typing import Tuple, Union
from sklearn.linear_model import LogisticRegression

XY = Tuple[np.ndarray, np.ndarray]
LogRegParams = Union[XY, Tuple[np.ndarray]]

def get_model_parameters(model):
    if model.fit_intercept:
        return (model.coef_, model.intercept_)
    return (model.coef_,)

def set_model_params(model: LogisticRegression, params: LogRegParams) -> LogisticRegression:
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def set_initial_params(model: LogisticRegression, n_features: int):
    model.classes_ = np.array([0, 1])
    model.coef_ = np.zeros((1, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((1,))

def load_custom_data(path: str) -> XY:
    df = pd.read_csv(path).dropna()
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    return X, y
