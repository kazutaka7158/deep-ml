import math
import numpy as np

PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	# 2) Perform linear regression on the detrended data.
	# 3) Predict day 15's base consumption.
	# 4) Add the day 15 fluctuation back.
	# 5) Round, then add a 5% safety margin (rounded up).
	# 6) Return the final integer.
	
	consumption_data = np.array(consumption_data, dtype=float)
	for i in range(len(consumption_data)):
		fluc_term = 10 * math.sin(2*PI * (i + 1) / 10)
		consumption_data[i] = consumption_data[i] - fluc_term
	
	# Linear Regression with x is list indices, y is data. f(x) = wx + b
	sum_xy = 0
	sum_x = 0
	sum_y = 0
	sum_square_x = 0
	for i, y in enumerate(consumption_data):
		sum_xy += (i + 1) * y
		sum_x += (i + 1)
		sum_y += y
		sum_square_x += (i + 1) * (i + 1)

	n = len(consumption_data)
	w = (n * sum_xy - sum_x * sum_y) / (n * sum_square_x - sum_x * sum_x)
	b = (sum_y - w * sum_x )/ n

	# Prediction phase
	output = w * 15 + b
	output += 10 * math.sin(2*PI * 15 / 10)
	output = math.ceil(1.05 * output)

	return output
