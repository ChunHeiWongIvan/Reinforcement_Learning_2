import numpy as np

# Initialize a 3x3 array with random values from a standard normal distribution
arr = np.random.randn(3, 3)
print(arr)

# Initialize a 3x3 array with values from a normal distribution with mean 0 and standard deviation 1
arr = np.random.normal(0, 1, (4, 3))
print(arr)
