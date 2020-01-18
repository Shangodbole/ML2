import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import KFold
# from sklearn.metrics.regression import
import matplotlib.pyplot as plt

model = LogisticRegression()
arr = np.array([2, 3, 4, 5, 6])

X = np.array([[1, 2], [3, 4], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5], [4, 5]])
# y = np.ones(X.shape[0])
y = np.array([0, 1, 0, 0, 1, 1, 1, 0])
n_fold = KFold(n_splits=2)
kf = n_fold.get_n_splits(X, y)
print(n_fold)
for train_idx, test_idx in n_fold.split(X, y):
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    model.fit(X_train, y_train)

pred = model.predict([[6, 6]])
print(model.get_params())
print(pred)
print(model.coef_)
print(model.intercept_)