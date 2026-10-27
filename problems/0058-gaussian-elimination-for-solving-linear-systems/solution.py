import numpy as np

def gaussian_elimination(A, b):
	"""
	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
	:param A: Coefficient matrix
	:param b: Right-hand side vector
	:return: Solution vector x
	"""

	A = np.array(A, dtype=float)
	b = np.array(b, dtype=float)

	N = len(b)

	# Transformation phase
	for i in range(N):
		pivot = i + np.argmax(np.abs(A[i:, i]))
		if pivot != i:
			A[[i, pivot]] = A[[pivot, i]]
			b[[i, pivot]] = b[[pivot, i]]

		for j in range(i + 1, N):
			temp = A[j][i]
			A[j] = A[i] * temp / A[i][i] - A[j]
			b[j] = b[i] * temp / A[i][i] - b[j]

	# Backward substitution
	res = []
	res.append(b[N - 1] / A[N - 1][N - 1])

	for k in range(N - 2, -1, -1):
		cal = b[k]
		for l in range(k + 1, N):
			cal = cal - A[k][l] * res[l - k - 1]

		cal = cal / A[k][k]
		res.insert(0, cal)

	return np.array(res, dtype=float)
