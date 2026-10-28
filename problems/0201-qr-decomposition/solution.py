import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
	"""
	Perform QR decomposition using Gram-Schmidt process.
	
	Args:
		A: An m x n matrix represented as list of lists
	
	Returns:
		Tuple of (Q, R) where Q is orthogonal and R is upper triangular
	"""
	A = np.array(A, dtype=float)
	m = len(A)
	n = len(A[0])

	Q = np.zeros_like(A, dtype=float)
	R = np.zeros(shape=(n, n))

	for j in range(n):
		v = A[:, j].copy()
		for i in range(j):
			R[i][j] = Q[:, i] @ A[:, j]
			v -= R[i][j] * Q[:, i]
		l2_norm = np.linalg.norm(v)
		R[j, j] = l2_norm
		Q[:, j] = v / l2_norm

	return Q, R
