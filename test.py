import matplotlib.pyplot as plt

# Example plot
plt.plot([1, 2, 3], [4, 5, 6])

# Adding labels with subscripts
plt.xlabel(r'$\text{Time (s)}$')
plt.ylabel(r'$\text{Speed (m/s)}$')

# Adding a title with subscripts
plt.title(r'Plot of $y = x_1 + x_2$')

# Show the plot
plt.show()
