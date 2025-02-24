import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import radiation_discrete
import particle_filter as pf

# Initialising environment

# Initialise search area/ possible source strength
search_area_x = 50
search_area_y = 50
max_radiation_level = 500

# Initalising source parameters
source_x = 25
source_y = 40
radiation_level = 100
sd_noise = 0.01 # Sensor measurement noise
source = radiation_discrete.source(source_x, source_y, radiation_level, sd_noise)

# Initialising agent parameters
agent_x = 10
agent_y = 10
agent_moveDist = 1
agent = radiation_discrete.agent(agent_x, agent_y, search_area_x, search_area_y, agent_moveDist)
agent_positions = [] # Save past agent positions

# Create clipped array with mesh of radiation level
rad_x = np.linspace(0, 50, 100)
rad_y = np.linspace(0, 50, 100)
rad_X, rad_Y = np.meshgrid(rad_x, rad_y)
rad_Z = source.radiation_level(rad_X, rad_Y)
Z_clipped = np.clip(rad_Z, 0, 100)

# Define the action set
actions = [agent.moveUp, agent.moveDown, agent.moveLeft, agent.moveRight]

# Initialise particle filter

N = 1000 # Number of particles

particles = pf.create_uniform_particles((0, search_area_x), (0, search_area_y), (0, max_radiation_level), N) # Initialise particles randomly in search area

weights = np.ones(N) / N # All weights initalised equally

print(pf.weighted_particle_sample(particles, weights))