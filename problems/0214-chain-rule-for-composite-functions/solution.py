import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	n = len(functions)

	# Forward pass
	u = [x]
	for i in range(n - 1, -1, -1):
		if functions[i] == "square":
			u.append(u[n - i - 1] * u[n - i - 1])
		if functions[i] == "sin":
			u.append(np.sin(u[n - i - 1]))
		if functions[i] == "cos":
			u.append(np.cos(u[n - i - 1]))
		if functions[i] == "exp":
			u.append(np.exp(u[n - i - 1]))
		if functions[i] == "log":
			u.append(np.log(u[n - i - 1]))

	# Backpropagation
	du = []
	for i in range(n - 1, -1, -1):
		if functions[i] == "square":
			du.append(2 * u[n - i - 1])
		if functions[i] == "sin":
			du.append(np.cos(u[n - i - 1]))
		if functions[i] == "cos":
			du.append(-np.sin(u[n - i - 1]))
		if functions[i] == "exp":
			du.append(np.exp(u[n - i - 1]))
		if functions[i] == "log":
			du.append(1 / u[n - i - 1])

	res = 1
	for d in du:
		res *= d
	return res
