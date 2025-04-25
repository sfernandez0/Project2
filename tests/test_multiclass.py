# tests/test_multiclass_gbm.py
import importlib
import numpy as np
import os
import sys
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ------------------------------------------------------------------
# Adjust this import so that it points to *your* implementation
# ------------------------------------------------------------------
gbc_mod = importlib.import_module('model.gradient_boosting_classifier')
GradientBoostingClassifier = getattr(gbc_mod, 'GradientBoostingClassifier')

@pytest.fixture(scope="module")
def iris_data():
    """Return a stratified train/test split of the Iris data set."""
    X, y = load_iris(return_X_y=True)
    return train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )


def test_fit_and_predict_multiclass(iris_data):
    """Model trains, predicts, and produces well-formed probabilities."""
    X_train, X_test, y_train, y_test = iris_data

    gbm = GradientBoostingClassifier(
        task="classification",
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=5,
    )
    gbm.fit(X_train, y_train)

    # -------- accuracy on held-out data --------
    y_pred = gbm.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    assert accuracy >= 0.90, f"Accuracy too low: {accuracy:.3f}"

    # -------- probability matrix --------
    proba = gbm.predict_proba(X_test)

    n_classes = len(np.unique(y_train))
    # shape must be (n_samples, n_classes)
    assert proba.shape == (X_test.shape[0], n_classes)

    # each row must sum to 1 (soft-max property)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_classes_attribute(iris_data):
    """After fitting, `classes_` must match the unique training labels."""
    X_train, _, y_train, _ = iris_data
    gbm = GradientBoostingClassifier(task="classification")
    gbm.fit(X_train, y_train)

    assert np.array_equal(np.sort(gbm.classes_), np.unique(y_train))


def test_predict_proba_raises_on_regression():
    """`predict_proba` should raise AttributeError in regression mode."""
    X = np.random.randn(20, 4)
    y = np.random.randn(20)

    gbm = GradientBoostingClassifier(task="regression", n_estimators=10)
    gbm.fit(X, y)

    with pytest.raises(AttributeError):
        _ = gbm.predict_proba(X)
