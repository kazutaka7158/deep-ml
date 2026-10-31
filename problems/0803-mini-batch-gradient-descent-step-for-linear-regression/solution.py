import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    m = len(batch_indices)

    weights_grad = X[batch_indices].T @ (X[batch_indices] @ weights + bias - y[batch_indices])
    weights_grad = weights_grad * (2 / m)

    bias_grad = np.sum((X[batch_indices] @ weights + bias - y[batch_indices])) * (2 / m)

    weights = weights - lr * weights_grad
    bias = bias - lr * bias_grad

    res = np.append(weights, bias)
    return res