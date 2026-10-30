import numpy as np

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
	"""
	Compute partial derivatives of multivariable functions.
	
	Args:
		func_name: Function identifier
			'poly2d': f(x,y) = x²y + xy²
			'exp_sum': f(x,y) = e^(x+y)
			'product_sin': f(x,y) = x·sin(y)
			'poly3d': f(x,y,z) = x²y + yz²
			'squared_error': f(x,y) = (x-y)²
		point: Point (x, y) or (x, y, z) at which to evaluate
	
	Returns:
		Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
	"""
	if func_name == "poly2d":
		dx = 2 * point[0] * point[1] + point[1] * point[1]
		dy = 2 * point[0] * point[1] + point[0] * point[0]
		return (dx, dy)
	if func_name == "exp_sum":
		return (np.exp(point[0] + point[1]), np.exp(point[0] + point[1]))
	if func_name == "product_sin":
		dx = np.sin(point[1])
		dy = point[0] * np.cos(point[1])
		return (dx, dy)
	if func_name == "poly3d":
		dx = 2 * point[0] * point[1]
		dy = point[0]**2 + point[2]**2
		dz = 2 * point[1] * point[2]
		return (dx, dy, dz)
	dx = 2 * (point[0] - point[1])
	dy = 2 * (point[1] - point[0])
	return (dx, dy)
