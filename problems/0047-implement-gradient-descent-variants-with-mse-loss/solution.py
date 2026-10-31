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

    for i in range(n_epochs):
        preds = X @ weights

        loss = np.linalg.norm(preds - y)
        loss = loss / m
        if method == "batch":
            gradients = X.T @ (preds - y)
            gradients = gradients * (2 / m)
            weights = weights - learning_rate * gradients

        if method == "stochastic":
            for k in range(m):
                single_grad = 2 * (X[k] @ weights - y[k]) * X[k].T
                weights = weights - learning_rate * single_grad

        if method == "mini_batch":
            num_batchs = m // batch_size
            if num_batchs * batch_size < m:
                num_batchs += 1
            for k in range(num_batchs):
                batch_grad = X[k * batch_size : (k + 1) * batch_size, :] @ weights
                batch_grad = batch_grad - y[k * batch_size : ((k + 1) * batch_size)]
                batch_grad = X[k * batch_size : (k + 1) * batch_size, :].T @ batch_grad

                weights = weights - learning_rate * (2/batch_size) * batch_grad
    return weights
