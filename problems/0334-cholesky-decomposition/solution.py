import numpy as np

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    
    Args:
        A: A symmetric positive-definite matrix (2D list or numpy array)
    
    Returns:
        L: Lower triangular matrix such that A = L @ L.T as a 2D list,
           or -1 if decomposition is not possible
    """
    # Your code here

    A = np.array(A, dtype=float)

    # Check if A is a square matrix
    if len(A) != len(A[0]):
        return -1

    # Check if A is symmetric
    A_transpose = np.transpose(A)
    for i in range(len(A)):
        for j in range(i, len(A)):
            if A_transpose[i][j] != A[j][i]:
                return -1

    L = np.zeros_like(A)
    N = len(A)

    for i in range(N):
        for j in range(N):
            # Calculate diagonal elements
            if i == j:
                sum_pre_L = 0
                for k in range(0, i):
                    sum_pre_L += L[i][k] * L[i][k]
                sqrt_term = A[i][i] - sum_pre_L
                if sqrt_term <= 0:
                    return -1
                L[i][i] = np.sqrt(sqrt_term)                

            # Calculate off-diagonal elements
            if i > j:
                sum_pre_L = 0
                for k in range(0, j):
                    sum_pre_L += L[i][k] * L[j][k]
                L[i][j] = (A[i][j] - sum_pre_L) / L[j][j]

    return L.tolist()
