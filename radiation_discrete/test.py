import numpy as np
import matplotlib.pyplot as plt

x = [0, 1, 2]
y = [10, 13, 11]
y = np.array(y)    

offset = 5

plt.plot(x, y+offset)
plt.show()