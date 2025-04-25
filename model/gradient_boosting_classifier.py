import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.decision_tree_regressor import DecisionTreeRegressor


class GradientBoostingClassifier:
    """
    Gradient Boosting supporting both classification and regression.
    Use task='classification' for binary logistic loss,
    and task='regression' for squared error loss.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, min_samples_split=2,
                 task='classification'):
        if task not in ('classification', 'regression'):
            raise ValueError("task must be 'regression' or 'classification'")
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.task = task

    @staticmethod
    def _softmax(F):
        """Fila-wise softmax; F shape = (n_samples, K)"""
        e = np.exp(F - F.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    

    def _sigmoid(self, f):
        return 1.0 / (1.0 + np.exp(-f))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_samples_, self.n_features_ = X.shape

        if self.task == "regression":
            self.f0_ = float(np.mean(y))
            self.trees_ = []
            F = np.full_like(y, self.f0_, dtype=float)

            for _ in range(self.n_estimators):
                residuals = y - F
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth, min_samples_split=self.min_samples_split
                )
                tree.fit(X, residuals)
                update = tree.predict(X)
                F += self.lr * update
                self.trees_.append(tree)
            return self  # <-- finished

        # ---------- classification ----------
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.K_ = len(self.classes_)
        self.trees_ = [] 

        if self.K_ == 2:
            # ---------- binary case  ----------
            p0 = np.clip(y.mean(), 1e-15, 1 - 1e-15)
            self.f0_ = np.log(p0 / (1 - p0))
            F = np.full(self.n_samples_, self.f0_, dtype=float)

            for _ in range(self.n_estimators):
                p = self._sigmoid(F)
                residuals = y - p
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth, min_samples_split=self.min_samples_split
                )
                tree.fit(X, residuals)
                F += self.lr * tree.predict(X)
                self.trees_.append([tree]) 
            return self

        # ---------- multiclass (K > 2) ----------
        Y_bool = np.zeros((self.n_samples_, self.K_), dtype=float)
        Y_bool[np.arange(self.n_samples_), y_idx] = 1

        class_priors = Y_bool.mean(axis=0).clip(1e-15, 1 - 1e-15)
        self.f0_ = np.log(class_priors)  # shape (K,)
        F = np.full((self.n_samples_, self.K_), self.f0_, dtype=float)

        for m in range(self.n_estimators):
            P = self._softmax(F)  # shape (n, K)
            residuals = Y_bool - P 

            trees_m = []
            for k in range(self.K_):
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth, min_samples_split=self.min_samples_split
                )
                tree.fit(X, residuals[:, k])
                update = tree.predict(X)
                F[:, k] += self.lr * update
                trees_m.append(tree)

            self.trees_.append(trees_m)

        return self
    
    def _decision_function(self, X):
        X = np.asarray(X)
        if self.task == "regression":
            F = np.full(X.shape[0], self.f0_, dtype=float)
            for tree in self.trees_:
                F += self.lr * tree.predict(X)
            return F

        if self.K_ == 2:
            F = np.full(X.shape[0], self.f0_, dtype=float)
            for trees_m in self.trees_:
                tree = trees_m[0]
                F += self.lr * tree.predict(X)
            return F  # shape (n,)

        # multiclass
        F = np.tile(self.f0_, (X.shape[0], 1))
        for trees_m in self.trees_:
            for k, tree in enumerate(trees_m):
                F[:, k] += self.lr * tree.predict(X)
        return F  # shape (n, K)

    def predict_proba(self, X):
        if self.task != "classification":
            raise AttributeError("predict_proba only exist for clasification")
        F = self._decision_function(X)
        if self.K_ == 2:
            p1 = self._sigmoid(F)
            return np.column_stack([1 - p1, p1])
        return self._softmax(F)

    def predict(self, X):
        if self.task == "regression":
            return self._decision_function(X)
        proba = self.predict_proba(X)
        class_idx = proba.argmax(axis=1)
        return self.classes_[class_idx]