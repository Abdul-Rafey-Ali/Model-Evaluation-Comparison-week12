"""Model training and evaluation for Lab 11.

Computes Accuracy, Precision, Recall, F1 for:
- BernoulliNB
- RandomForestClassifier
- GaussianNB
- DecisionTreeClassifier
- MultinomialNB
- KNeighborsClassifier

Uses train_test_split with shuffle=False.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def _weighted_precision_recall_f1(y_true, y_pred) -> Dict[str, float]:
    labels = np.unique(y_pred)

    precision = metrics.precision_score(
        y_true, y_pred, average="weighted", labels=labels, zero_division=0
    )
    recall = metrics.recall_score(
        y_true, y_pred, average="weighted", labels=labels, zero_division=0
    )
    f1 = metrics.f1_score(
        y_true, y_pred, average="weighted", labels=labels, zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def train_and_evaluate(X, y) -> Dict[str, Dict[str, float]]:
    # Lab 11: train/test split (shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    results: Dict[str, Dict[str, float]] = {}

    models = {
        "Bernoulli": BernoulliNB(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gaussian": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Multinomial": MultinomialNB(),
        "KNeighbors": KNeighborsClassifier(),
    }

    # Some NB variants expect non-negative values; normalize/shift.
    X_train_nb = X_train.copy()
    X_test_nb = X_test.copy()

    min_val = np.nanmin(X_train_nb.values)
    if min_val < 0:
        shift = abs(min_val)
        X_train_nb = X_train_nb + shift
        X_test_nb = X_test_nb + shift

    # BernoulliNB expects binary-ish data; we binarize with a simple threshold.
    bern_train = (X_train_nb.values > 0).astype(int)
    bern_test = (X_test_nb.values > 0).astype(int)

    for name, model in models.items():
        if name == "Bernoulli":
            model.fit(bern_train, y_train)
            y_pred = model.predict(bern_test)
        elif name == "Multinomial":
            # MultinomialNB requires non-negative counts; we use abs and cast.
            multi_train = np.abs(X_train_nb.values)
            multi_test = np.abs(X_test_nb.values)
            model.fit(multi_train, y_train)
            y_pred = model.predict(multi_test)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        m = _weighted_precision_recall_f1(y_test, y_pred)
        results[name] = {"accuracy": float(accuracy), **m}

    return results

