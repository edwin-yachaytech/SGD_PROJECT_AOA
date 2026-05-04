"""
Final project: Stochastic Gradient Descent for Logistic Regression.

This compact script contains:
- WDBC data loading
- train/test split
- standardization
- logistic regression trained with Batch GD, SGD, and Mini-Batch SGD
- metrics
- runtime and theoretical cost accounting

Author: Edwin Fabricio Quizhpe Aguilar
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
DATA_PATH = Path("wdbc.data")


FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]


def load_wdbc(path: Path):
    """Load WDBC data. Label encoding: malignant=1, benign=0."""
    columns = ["id", "diagnosis", *FEATURE_NAMES]
    df = pd.read_csv(path, header=None, names=columns)
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = (df["diagnosis"] == "M").astype(float).to_numpy()
    return X, y, df


def train_test_split(X, y, test_ratio=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(round(len(y) * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_train_test(X_train, X_test):
    """Fit standardization on training data only to avoid data leakage."""
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma = np.where(sigma == 0.0, 1.0, sigma)
    return (X_train - mu) / sigma, (X_test - mu) / sigma


def sigmoid(z):
    """Numerically stable sigmoid."""
    z = np.asarray(z)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def binary_cross_entropy(y_true, p_pred, eps=1e-15):
    p = np.clip(p_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def classification_metrics(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return accuracy, precision, recall, f1, tp, tn, fp, fn


class LogisticOptimizer:
    """Logistic regression trained from scratch with first-order methods."""

    def __init__(
        self,
        method="sgd",
        learning_rate=0.05,
        epochs=120,
        batch_size=32,
        lr_schedule="constant",
        decay=0.002,
        l2=0.001,
        seed=SEED,
    ):
        self.method = method
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.decay = decay
        self.l2 = l2
        self.seed = seed
        self.weights = None
        self.bias = 0.0
        self.loss_history = []
        self.update_count = 0
        self.fit_time = 0.0

    def lr(self, step):
        if self.lr_schedule == "time_decay":
            return self.learning_rate / (1.0 + self.decay * step)
        return self.learning_rate

    def predict_proba(self, X):
        return sigmoid(X @ self.weights + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def loss(self, X, y):
        base = binary_cross_entropy(y, self.predict_proba(X))
        penalty = 0.5 * self.l2 * float(np.dot(self.weights, self.weights))
        return base + penalty

    def batch_gradient(self, Xb, yb):
        p = sigmoid(Xb @ self.weights + self.bias)
        error = p - yb
        grad_w = (Xb.T @ error) / len(yb) + self.l2 * self.weights
        grad_b = float(np.mean(error))
        return grad_w, grad_b

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        self.update_count = 0
        start = time.perf_counter()

        for _ in range(self.epochs):
            if self.method == "batch":
                grad_w, grad_b = self.batch_gradient(X, y)
                eta = self.lr(self.update_count)
                self.weights -= eta * grad_w
                self.bias -= eta * grad_b
                self.update_count += 1
            else:
                idx = rng.permutation(n_samples)
                step = 1 if self.method == "sgd" else self.batch_size
                for start_idx in range(0, n_samples, step):
                    batch_idx = idx[start_idx:start_idx + step]
                    grad_w, grad_b = self.batch_gradient(X[batch_idx], y[batch_idx])
                    eta = self.lr(self.update_count)
                    self.weights -= eta * grad_w
                    self.bias -= eta * grad_b
                    self.update_count += 1
            self.loss_history.append(self.loss(X, y))

        self.fit_time = time.perf_counter() - start
        return self


def cost_per_update(method, n, d, batch_size=1):
    if method == "sgd":
        return d
    if method == "mini_batch":
        return batch_size * d
    if method == "batch":
        return n * d
    raise ValueError(method)


def run():
    if not DATA_PATH.exists():
        raise FileNotFoundError("wdbc.data must be in the same folder as run_sgd_final.py")

    X, y, df = load_wdbc(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_test = standardize_train_test(X_train, X_test)
    n, d = X_train.shape

    print("=" * 72)
    print("Stochastic Gradient Descent Final Project")
    print("=" * 72)
    print(f"Dataset: WDBC | instances={len(df)} | features={d}")
    print(f"Class counts: malignant={(df['diagnosis'] == 'M').sum()} | benign={(df['diagnosis'] == 'B').sum()}")
    print(f"Train/Test split: {len(y_train)} train | {len(y_test)} test")
    print()

    configs = [
        ("Batch GD", "batch", 0.08, 1, "constant"),
        ("SGD", "sgd", 0.03, 1, "time_decay"),
        ("Mini-Batch SGD", "mini_batch", 0.08, 32, "time_decay"),
    ]

    print("Method              Accuracy  Precision  Recall   F1       Test BCE  Time(s)  Updates  Cost/update")
    print("-" * 100)
    for label, method, lr, batch_size, schedule in configs:
        model = LogisticOptimizer(
            method=method,
            learning_rate=lr,
            batch_size=batch_size,
            lr_schedule=schedule,
        ).fit(X_train, y_train)

        probs = model.predict_proba(X_test)
        preds = model.predict(X_test)
        acc, prec, rec, f1, tp, tn, fp, fn = classification_metrics(y_test, preds)
        bce = binary_cross_entropy(y_test, probs)
        cost = cost_per_update(method, n, d, batch_size)

        print(
            f"{label:<19} "
            f"{acc:>8.4f}  {prec:>9.4f}  {rec:>6.4f}  {f1:>7.4f}  "
            f"{bce:>8.4f}  {model.fit_time:>7.4f}  {model.update_count:>7}  {cost:>11}"
        )

    print()
    print("Complexity summary:")
    print("SGD update: O(d), because one dense update touches d features and d weights.")
    print("Mini-Batch SGD update: O(bd), because it processes b samples before updating.")
    print("Batch GD update: O(nd), because it processes all n samples before updating.")


if __name__ == "__main__":
    run()
