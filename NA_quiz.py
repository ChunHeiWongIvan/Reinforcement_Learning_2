import numpy as np
import matplotlib.pyplot as plt

# Data points
x = np.array([0, 5, 10, 15, 20])
y = np.array([9, 15, 22, 30, 44])

# Perform least squares fit (linear fit)
A = np.vstack([x, np.ones_like(x)]).T  # Create design matrix
m, c = np.linalg.lstsq(A, y, rcond=None)[0]  # Solve for slope (m) and intercept (c)

# Print the slope and intercept
m, c
