import numpy as np
import torch
from numpy.random import uniform

# 1. Initialise particles randomly in search region

def create_uniform_particles(x_range, y_range, conc_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(conc_range[0], conc_range[1], size=N)
    return particles

# Can be used for resampling as it creates spread of particles around likely estimate
def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (np.random.randn(N) * std[0])
    particles[:, 1] = mean[1] + (np.random.randn(N) * std[1])
    particles[:, 2] = mean[2] + (np.random.randn(N) * std[2])
    return particles

# 2. No need for predict step as the source is static (position and strength)

def likelihood(agent_x, agent_y, particles, measured, sd_noise_pct):
    
    likelihood = np.empty(particles.shape[0])

    for i, particle in enumerate(particles):

        # Compute radiation level at the sensor's position for given particle state (model for radiation level vs. distance)
        dist = np.sqrt((agent_x-particle[0])**2 + (agent_y-particle[1])**2)
        radiation_level = particle[2] / dist**2

        # Compute standard deviation of noise (scales with detected radiation level)
        sigmaN = sd_noise_pct * radiation_level

        # Compute likelihood of source being at the particle with radiation level
        likelihood[i] = (1/((sigmaN)*np.sqrt(2*np.pi)))*np.exp(-((measured-radiation_level)**2)/(2*(sigmaN)**2))

    return likelihood


def update_weights(weights_old, likelihood):

    # Compute new weights by multiplying old weights with likelihood of each particle
    weights_new = weights_old * likelihood

    # Avoid round-off to zero
    weights_new += 1.e-300

    # Normalise weights to ensure weights sum to 1 to represent distribution
    return weights_new / np.sum(weights_new)
    
def estimate(particles, weights):
    # Convert particles and weights to torch tensors (if they aren't already)
    particles = torch.tensor(particles, dtype=torch.float32)
    weights = torch.tensor(weights, dtype=torch.float32)

    # Returns mean and variance of weighted particles
    pos = particles[:, 0:3]
    
    # Calculate the weighted mean
    mean = torch.sum(pos * weights.unsqueeze(1), dim=0) / torch.sum(weights)
    
    # Calculate the weighted variance
    var = torch.sum(((pos - mean)**2) * weights.unsqueeze(1), dim=0) / torch.sum(weights)
    
    return mean, var

def resampling(particles, weights):

    N = np.size(weights)

    # Compute ESS (Effective sample size) to determine how well the particles represent the distribution
    ess = 1 / np.sum(weights**2)

    if ess < 0.5*N:
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1. # avoid round-off error by setting last element of cumulative_sum array to 1.0
        indexes = np.searchsorted(cumulative_sum, np.random.random(N))

        # resample according to indexes
        sd_noise = np.array([0.1, 0.1, 0.5])  # Different standard deviations for each dimension
        particles_new = particles[indexes] + np.random.normal(loc=0, scale=sd_noise, size=particles[indexes].shape)

        weights_new = np.ones(N) / N

        return particles_new, weights_new
    
    return particles, weights

def weighted_particle_sample(particles, weights):

    cumulative_sum = np.cumsum(weights) # Computes cumulative distribution of weights
    cumulative_sum[-1] = 1. # avoid round-off error by setting last element of cumulative_sum array to 1.0

    index = np.searchsorted(cumulative_sum, np.random.random()) # Finds index of first particle whose CDF value >= random number from 0 to 1

    return particles[index]
