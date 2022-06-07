from typing import Iterable

from sklearn.model_selection import train_test_split


def my_stratified_train_test_split(data: Iterable, label_func, **kwargs):
    X = data
    y = [label_func(x) for x in data]
    X_train, X_test, _, _ = train_test_split(X, y, stratify=y, **kwargs)
    return X_train, X_test