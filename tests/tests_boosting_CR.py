import importlib
import os
import sys

import numpy as np

# Ensure the project root is on sys.path so we can import model.gradient_boosting_classifier
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# Path to your CSV data folder
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def load_csv(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

# Dynamically import the GradientBoostingClassifier
gbc_mod = importlib.import_module('model.gradient_boosting_classifier')
Model = getattr(gbc_mod, 'GradientBoostingClassifier')

def test_regression_csv():
    # regression.csv lives in project_root/data
    path = os.path.join(DATA_DIR, 'regression.csv')
    X, y = load_csv(path)

    model = Model(task='regression',
                  n_estimators=100,
                  learning_rate=0.1,
                  max_depth=3,
                  min_samples_split=2)
    model.fit(X, y)
    preds = model.predict(X)
    mse = np.mean((preds - y)**2)
    assert mse < 2.0, f"MSE demasiado alto: {mse:.4f}"

def test_classification_csv():
    # classification.csv lives in project_root/data
    path = os.path.join(DATA_DIR, 'classification.csv')
    X, y = load_csv(path)
    y = y.astype(int)

    model = Model(task='classification',
                  n_estimators=100,
                  learning_rate=0.1,
                  max_depth=3,
                  min_samples_split=2)
    model.fit(X, y)
    preds = model.predict(X)
    acc = np.mean(preds == y)
    assert acc > 0.8, f"Accuracy demasiado baja: {acc:.2%}"
