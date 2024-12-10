import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import radiation

# Initialise search area
search_area_x = 50.0
search_area_y = 50.0

# Initalising source parameters
source_x = 25.00
source_y = 40.00
radiation_level = 100.00
source = radiation.source(source_x, source_y, radiation_level)

# Initialising agent parameters
agent_x = 25.00
agent_y = 10.00
agent_moveDist = 1.00
agent = radiation.agent(agent_x, agent_y, search_area_x, search_area_y, agent_moveDist)
agent_positions = [] # Save past agent positions

# Define the action set
actions = [agent.moveUp, agent.moveDown, agent.moveLeft, agent.moveRight]

# Create clipped array with mesh of radiation level
rad_x = np.linspace(0, 50, 100)
rad_y = np.linspace(0, 50, 100)
rad_X, rad_Y = np.meshgrid(rad_x, rad_y)
rad_Z = source.radiation_level(rad_X, rad_Y)
Z_clipped = np.clip(rad_Z, 0, 100)

# Show agent as green point
for i in range(100):
    random.choice(actions)()  # Agent makes a random move
    
    # Clear the figure or axes to remove previous markers
    plt.clf()  # Clear the figure
    
    # Re-plot the radiation map and radiation source
    plt.contourf(rad_X, rad_Y, Z_clipped, 200, cmap='viridis')
    plt.colorbar()
    plt.plot(source.x(), source.y(), 'ro', markersize=4)
    
    # Show agent as green point (current position only)
    plt.plot(agent.x(), agent.y(), marker='x', color=(0.2, 0.8, 0), markersize=4)
    
    # Pause briefly to update the plot
    plt.xlim(0, search_area_x)
    plt.ylim(0, search_area_y)
    plt.pause(0.1)

# Plot graph according to search area and show graph
plt.show()