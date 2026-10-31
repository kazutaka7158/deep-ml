import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    weights = np.array(weights, dtype=float)

    m, n = X.shape

    if method == "batch":
        step = m
    if method == "stochastic":
        step = 1
    if method == "mini_batch":
        step = batch_size

    for _ in range(n_epochs):
        for start in range(0, m, step):
            X_b = X[start:start + step]
            y_b = y[start:start + step]
            grad = (2 / len(y_b)) * X_b.T @ (X_b @ weights - y_b)

            weights = weights - learning_rate * grad
    return weights
