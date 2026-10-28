import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
	"""
	Solve the system Ax = b using the Conjugate Gradient method.

	:param A: Symmetric positive-definite matrix
	:param b: Right-hand side vector
	:param n: Maximum number of iterations
	:param x0: Initial guess for solution (default is zero vector)
	:param tol: Convergence tolerance
	:return: Solution vector x
	"""
	A = np.array(A, dtype=float)
	b = np.array(b, dtype=float)

	x = np.zeros_like(b)
	if x0 is not None:
		x = x0
	p = r = b - A @ x

	for i in range(n):
		alpha = (r.T @ r) / (p.T @ A @ p)
		
		x = x + alpha * p
		r_new = r - alpha * A @ p

		if np.linalg.norm(r_new) < tol:
			return x

		beta = (r_new.T @ r_new) / (r.T @ r)
		p = r_new + beta * p
		r = r_new

	return x
