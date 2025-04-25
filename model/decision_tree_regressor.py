import numpy as np

class _Node:
    def __init__(self, depth, max_depth, min_samples_split):
        self.depth, self.max_depth = depth, max_depth
        self.min_samples_split = min_samples_split
        self.left = self.right = None
        self.feature = None
        self.threshold = None
        self.value = None

    def fit(self, X, y):
        # stopping
        if (self.depth >= self.max_depth or 
            X.shape[0] < self.min_samples_split):
            self.value = y.mean()
            return

        best = {'mse': np.inf}
        for j in range(X.shape[1]):
            for t in np.unique(X[:,j]):
                mask = X[:,j] <= t
                if mask.sum() < self.min_samples_split or (~mask).sum() < self.min_samples_split:
                    continue
                yL, yR = y[mask], y[~mask]
                mse = (yL.var()*yL.size + yR.var()*yR.size)/X.shape[0]
                if mse < best['mse']:
                    best.update(mse=mse, feature=j, thr=t, mask=mask)

        if best['mse'] == np.inf:
            self.value = y.mean()
            return

        self.feature, self.threshold = best['feature'], best['thr']
        self.left = _Node(self.depth+1, self.max_depth, self.min_samples_split)
        self.right = _Node(self.depth+1, self.max_depth, self.min_samples_split)
        self.left.fit(X[best['mask']], y[best['mask']])
        self.right.fit(X[~best['mask']], y[~best['mask']])

    def predict_one(self, x):
        if self.value is not None:
            return self.value
        if x[self.feature] <= self.threshold:
            return self.left.predict_one(x)
        else:
            return self.right.predict_one(x)

class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.root = _Node(0, max_depth, min_samples_split)

    def fit(self, X, y):
        self.root.fit(X, y)

    def predict(self, X):
        return np.array([self.root.predict_one(x) for x in X])
