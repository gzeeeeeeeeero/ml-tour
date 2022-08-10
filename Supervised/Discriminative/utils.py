from pandas import *
from numpy  import *

import matplotlib.pyplot as plt
import seaborn as sns

intercept = lambda X : column_stack((ones(X.shape[0]), X))

def fit(X, y):
    X = intercept(X)
    return (linalg.inv(X.T @ X) @ X.T @ y).reshape(1, -1)
def fit_dual(X, y): # used when features > samples
    X = intercept(X)
    return (X.T @ linalg.inv(X @ X.T) @ y).reshape(1, -1)

# OLS with L2 Regularization
def fit_L2(X, y, λ):
    X = intercept(X)
    return (linalg.inv(X.T @ X + λ) @ X.T @ y).reshape(1, -1)
def fit_L2(X, y, λ):
    X = intercept(X)
    return (X.T @ linalg.inv(X @ X.T + λ) @ y).reshape(1, -1)

def predict(X, Θ):
    X = intercept(X)
    return (fromiter((Θ @ x for x in X), dtype=float)).reshape(-1, 1)

def make_poly(X, degree):
    return column_stack(tuple([X]+[X**i for i in range(2, degree + 1)]))
