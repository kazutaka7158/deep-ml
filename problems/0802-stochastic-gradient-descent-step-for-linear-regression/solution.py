import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    n = X.shape[0]

    for step in range(n_iter):
        i = step % n
        grad = 2 * X[i].T * (X[i] @ weights - y[i])
        weights = weights - learning_rate * grad
    return weights