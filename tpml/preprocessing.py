import numpy as np
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from multipledispatch import dispatch
from numpy import ndarray
from .types import Results
from typing import Tuple


def unstandardise(X, mu, sigma) -> ndarray:
    """
    Reproject data back onto its original scale.
    Example
    -------
    >>> X = unstandardise(Xtransform, Xmean, Xstd)
    """
    return (X*sigma)+mu


@dispatch(ndarray)
def standardise(X: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Standardise a dataset column-wise to unary Gaussian.
    Example
    -------
    >>> X = np.random.randn(10, 2)
    >>> Xtransform, Xmean, Xstd = standardise(X)

    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mu) / sigma, mu, sigma


@dispatch(ndarray, ndarray, ndarray)
def standardise(X: ndarray, mean: ndarray, std: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Standardise a dataset column-wise to unary Gaussian.
    Example
    -------
    >>> X = np.random.randn(10, 2)
    >>> Xtransform, Xmean, Xstd = standardise(X)

    """
    return (X - mean) / std, mean, std


def get_xy(df: DataFrame, target_name: str, standardise_data: bool = False, train_size: float = 1.0,
           seed: int = 123) -> Results:
    X = df.drop(target_name, axis=1).values
    y = df[target_name].values.reshape(-1, 1)
    if train_size < 1.:
        Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=seed, train_size=train_size)
    else:
        Xtr, ytr = X, y
        Xte, yte = None, None
    if standardise_data:
        Xtr, Xmean, Xstd = standardise(Xtr)
        if Xte is not None:
            Xte = standardise(Xte, Xmean, Xstd)
    return Results((Xtr, Xte, ytr, yte), (Xmean, Xstd))


