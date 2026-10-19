
"""Part 2, Task B: regression and classification model factories."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from .transform import build_preprocessor


class GDLinearRegression(BaseEstimator, RegressorMixin):
    """Linear regression trained using gradient descent."""

    def __init__(self, lr: float = 0.1, n_iter: int = 1000):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        if self.n_iter < 1 or self.lr <= 0:
            raise ValueError("n_iter and lr must be positive")

        self.n_features_in_ = X.shape[1]
        self.coef_ = np.zeros(self.n_features_in_, dtype=float)
        self.intercept_ = 0.0
        self.loss_history_ = []

        for _ in range(self.n_iter):
            predictions = X @ self.coef_ + self.intercept_
            errors = predictions - y

            self.loss_history_.append(float(np.mean(errors**2)))

            self.coef_ -= self.lr * (2 / len(y)) * (X.T @ errors)
            self.intercept_ -= self.lr * 2 * float(np.mean(errors))

        errors = X @ self.coef_ + self.intercept_ - y
        self.loss_history_.append(float(np.mean(errors**2)))

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if not hasattr(self, "coef_"):
            raise RuntimeError("Call fit before predict")

        return X @ self.coef_ + self.intercept_


def make_regressor(
    kind: str = "linear",
    degree: int = 2,
    alpha: float = 1.0,
):
    """Create a regression pipeline."""

    if kind == "dummy":
        estimator = DummyRegressor(strategy="mean")
    elif kind == "linear":
        estimator = Ridge(alpha=0.0)
    elif kind == "ridge":
        estimator = Ridge(alpha=alpha)
    elif kind == "poly":
        estimator = Ridge(alpha=alpha)
    else:
        raise ValueError(f"Unknown regressor: {kind}")

    poly_degree = degree if kind == "poly" else 1

    return Pipeline(
        steps=[
            ("prep", build_preprocessor(degree=poly_degree)),
            ("model", estimator),
        ]
    )


def make_classifier(
    kind: str = "logistic",
    C: float = 1.0,
    class_weight=None,
):
    """Create a classification pipeline with probability predictions."""

    if kind == "dummy":
        estimator = DummyClassifier(strategy="prior")
    elif kind == "logistic":
        estimator = LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=1000,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown classifier: {kind}")

    return Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            ("model", estimator),
        ]
    )
