import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	res = {}

	gradient = np.array(gradient, dtype=float)

	magnitude = np.linalg.norm(gradient)
	res["magnitude"] = magnitude

	if magnitude == 0:
		res["direction"] = [0] * len(gradient)
		res["descent_direction"] = [0] * len(gradient)
	else:
		res["direction"] = [x / magnitude for x in gradient]
		res["descent_direction"] = [-x / magnitude for x in gradient]

	return res
