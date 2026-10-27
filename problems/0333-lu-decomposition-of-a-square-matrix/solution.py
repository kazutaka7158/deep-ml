import numpy as np

def lu_decomposition(A: list) -> tuple:
	"""
	Perform LU decomposition on a square matrix using Doolittle's method.
	
	Args:
		A: Square matrix as a list of lists
	
	Returns:
		tuple: (L, U) where L is lower triangular with 1s on diagonal,
		       U is upper triangular, and A = L @ U
	"""
	# Your code here
	A = np.array(A, dtype=float)
	N = len(A)

	L = np.identity(N, dtype=float)
	U = np.zeros_like(A, dtype=float)

	for i in range(N):
		for j in range(N):

			# Calculate matrix L
			if i > j:
				dot_term = np.dot(L[i, :j], U[:j, j]) if j > 0 else 0
				L[i][j] = (A[i][j] - dot_term) / U[j][j]

			# Calculate matrix U
			else:
				dot_term = np.dot(L[i, :i], U[:i, j]) if i > 0 else 0
				U[i][j] = A[i][j] - dot_term

	return L, U

