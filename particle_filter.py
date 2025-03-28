# Particle filter code structure based on chp 12 of Kalman and Bayesian Filters in Python by Roger R Labbe Jr.
# 
# https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb

import numpy as np
import torch
import random
import math

search_area_x = 50 # float
search_area_y = 50 # float

# 1. Initialise particles randomly in search region

def create_uniform_particles(x_range, y_range, strength_range, no_sources_range, N):

    # Structure of array row (each row represents one particle):
    # (N, x_1, y_1, strength_1, ... , x_N, y_N, strength_N),  where N is the number of sources predicted by one particle

    particles = np.full((N, 3*no_sources_range[1] + 1), np.nan) # Initalise array and fill with nan values (in order to have blank values when no. of source estimate is less than the max number)

    for i in range(N):
        no_sources = random.randint(int(no_sources_range[0]), int(no_sources_range[1])) # Uniform randomly choose number of sources for one particle
        particles[i, 0] = no_sources

        for j in range(no_sources):
            particles[i, 3*j + 1] = random.uniform(x_range[0], x_range[1])
            particles[i, 3*j + 2] = random.uniform(y_range[0], y_range[1])
            particles[i, 3*j + 3] = random.uniform(strength_range[0], strength_range[1])

    return particles

# Debugging: Initialise cheat particles at correct location

def create_cheat_particles(no_sources, source1, source2, source3, no_sources_range, N):

    # Structure of array row (each row represents one particle):
    # (N, x_1, y_1, strength_1, ... , x_N, y_N, strength_N),  where N is the number of sources predicted by one particle

    particles = np.full((N, 3*no_sources_range[1] + 1), np.nan) # Initalise array and fill with nan values (in order to have blank values when no. of source estimate is less than the max number)

    particles[:] = [no_sources, source1.x(), source1.y(), source1.source_radioactivity, source2.x(), source2.y(), source2.source_radioactivity, source3.x(), source3.y(), source3.source_radioactivity, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    return particles

# 2. No need for predict step as the source is static (position and strength) (half-life of sources markedly exceeds duration of localization)

# 3. Compute likelihood of particle representing the true state (source parameters) given the observation (sensor reading), using posterior distribution

# Check distance between sources and send likelihood of predictions that predict sources within 5 m of each other to 0

def check_within_distance(coords, distance_threshold=5):
    coords = np.array(coords)
    
    # Iterate over each pair of points
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            # Calculate the Euclidean distance between the pair
            distance = np.linalg.norm(coords[i] - coords[j])
            
            # Check if the distance is less than or equal to the threshold
            if distance <= distance_threshold:
                return True  # Return True if any pair is within the threshold
    
    return False  # No pair found within the threshold

def coordinates_out_of_bounds(coordinates, lower_bound=0, upper_bound=search_area_x):
    # Check if any x or y coordinate is outside the range [lower_bound, upper_bound]
    out_of_bounds = np.any((coordinates[:, 0] < lower_bound) | (coordinates[:, 0] > upper_bound) |
                           (coordinates[:, 1] < lower_bound) | (coordinates[:, 1] > upper_bound))
    
    return out_of_bounds

def extract_source_coordinates(particle): # To get source coordinates in a 2d array from the particle
    source_coordinates = []
    num_sources = int(particle[0])  # The first element is the number of sources
        
    # Extract x and y for each source
    for i in range(num_sources):
        x = particle[1 + 3*i]  # x value is at 1 + 3*i
        y = particle[2 + 3*i]  # y value is at 2 + 3*i
        source_coordinates.append([x, y])
    
    return np.array(source_coordinates)

def likelihood(agent_x, agent_y, particles, measured, sd_noise_pct, min_radiation_level, max_radiation_level):
    
    likelihood = np.empty(particles.shape[0])

    for i, particle in enumerate(particles):

        total_radiation_level = 0 # Initalise radiation level total for each particle

        for j in range(int(particle[0])): # Iterate through each set of source parameters for each source up to particle[0] (which stores no. of sources)

            # Compute radiation level from each source at the sensor's position for given particle state (model for radiation level vs. distance), then sum together
            dist = np.sqrt((agent_x-particle[3*j + 1])**2 + (agent_y-particle[3*j + 2])**2)
            if dist < 1:
                total_radiation_level += particle[3*j + 3]
            else:
                total_radiation_level += particle[3*j + 3] / dist**2

        # Compute standard deviation of noise (scales with detected radiation level)

        sigmaN = max(sd_noise_pct * total_radiation_level, 1.e-300) # Adding small offset to avoid round-off to zero

        # Compute likelihood of source being at the particle with radiation level
        likelihood[i] = (1/(np.sqrt(2*np.pi*(sigmaN**2))))*np.exp(-((measured-total_radiation_level)**2)/(2*(sigmaN)**2))

        for j in range(int(particle[0])):
            if particle[3*j + 3] < min_radiation_level or particle[3*j + 3] > max_radiation_level: # If particle predicts invalid levels of radiation, prediction is invalidated
                likelihood[i] = 0

        two_d_particle = extract_source_coordinates(particle)


        if check_within_distance(two_d_particle, 5) or coordinates_out_of_bounds(two_d_particle): # If sources are within 5 m of each other, or prediction out of bounds then its invalid
            likelihood[i] = 0

    return likelihood

# 4. Update weights for particles depending on their likelihood

def update_weights(weights_old, likelihood):

    # Compute new weights by multiplying old weights with likelihood of each particle
    weights_new = weights_old * likelihood

    # Avoid round-off to zero
    weights_new += 1.e-300

    # Normalise weights to ensure weights sum to 1 to represent distribution
    return weights_new / np.sum(weights_new)

# 5. Compute source term estimate(s) with current particles

def estimate(particles, weights):
    
    max_no_sources = (particles.shape[1] - 1) / 3

    # Convert particles and weights to torch tensors (if they aren't already)
    particles = torch.tensor(particles, dtype=torch.float32)
    weights = torch.tensor(weights, dtype=torch.float32)

    # Find the mode (most common value) of the number of sources across all particles

    num_sources = particles[:, 0].int()  # First column is the number of sources (N)
    mode_no_sources = np.bincount(num_sources.numpy()).argmax()  # Find the mode using numpy's bincount function
    
    # Filter particles with the mode number of sources
    filtered_particles = particles[num_sources == mode_no_sources]
    filtered_weights = weights[num_sources == mode_no_sources]

    # Initialize lists for storing weighted means and variances
    mean_x_pos, mean_y_pos, mean_strength = [], [], []
    var_x_pos, var_y_pos, var_strength = [], [], []

    # Iterate over each source to compute weighted means and variances
    for source in range(mode_no_sources):
        x_vals = filtered_particles[:, 3 * source + 1]
        y_vals = filtered_particles[:, 3 * source + 2]
        strengths = filtered_particles[:, 3 * source + 3]

        # Compute weighted means
        mean_x = torch.sum(x_vals * filtered_weights) / torch.sum(filtered_weights)
        mean_y = torch.sum(y_vals * filtered_weights) / torch.sum(filtered_weights)
        mean_c = torch.sum(strengths * filtered_weights) / torch.sum(filtered_weights)

        # Compute weighted variances
        var_x = torch.sum(filtered_weights * (x_vals - mean_x) ** 2) / torch.sum(filtered_weights)
        var_y = torch.sum(filtered_weights * (y_vals - mean_y) ** 2) / torch.sum(filtered_weights)
        var_c = torch.sum(filtered_weights * (strengths - mean_c) ** 2) / torch.sum(filtered_weights)

        # Append results
        mean_x_pos.append(mean_x.item())
        mean_y_pos.append(mean_y.item())
        mean_strength.append(mean_c.item())
        var_x_pos.append(var_x.item())
        var_y_pos.append(var_y.item())
        var_strength.append(var_c.item())
    
    missing_values = int(max_no_sources - mode_no_sources)

    mean_x_pos += [np.nan] * missing_values
    mean_y_pos += [np.nan] * missing_values
    mean_strength += [np.nan] * missing_values
    var_x_pos += [np.nan] * missing_values
    var_y_pos += [np.nan] * missing_values
    var_strength += [np.nan] * missing_values

    return mean_x_pos, mean_y_pos, mean_strength, var_x_pos, var_y_pos, var_strength
    

# 6. Resample particles if effective sample size is too small to represent true distribution

# Simple regularized resampling

def resampling_simple(particles, weights, min_no_sources, max_no_sources, min_radiation_level, max_radiation_level, EPS_START, EPS_END, EPS_DECAY, steps_done):

    N = np.size(weights)

    # Compute ESS (Effective sample size) to determine how well the particles represent the distribution
    ess = 1 / np.sum(weights**2)

    EPS_END = 0.10 # Custom minimum eps so particles always maintain some kind of diversity

    EPS_DECAY = 40000 # Custom, lower eps_decay so that perturbance decays faster

    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

    need_resample = ess < 0.5*N # Check if resampling is needed using ESS

    no_sources_change_pct = 0.004*eps_threshold # 0.4% probability for number of sources belief to increase, and to decrease (chance decays over time)

    scramble_pct = 0.004*eps_threshold # 0.1% probability for sources location/ strength to be scrambled

    if need_resample:
        particles_new = particles # Initalise separate instance of particles array 

        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1. # avoid round-off error by setting last element of cumulative_sum array to 1.0
        indexes = np.searchsorted(cumulative_sum, np.random.random(N))

        particles_new = particles[indexes]

        unique_values = np.unique(particles_new[:, 0]).astype(int) # Find unique values in column 0 (no. of sources) and cast to integer

        grouped_particles = {} # Dictionary to store separate arrays for each unique value

        particles_result = np.array([]) # Empty array to store results of resampling

        first_append = True

        for value in unique_values:
            matching_particles = particles_new[particles_new[:, 0] == value]
            grouped_particles[value] = matching_particles[:, 1:value*3 + 1] # Only extract relevant columns for computation

        for no_sources, particle_array in grouped_particles.items():

            n_dim = no_sources*3 # Number of columns corresponds to number of sources multiplied by 3 (for x, y, strength)

            perturbations = np.random.normal(0, 0.5*eps_threshold, (len(particle_array), n_dim)) # Perturbations decay over time as estimate gets more confident
    
            grouped_particles[no_sources] = particle_array + perturbations

            if grouped_particles[no_sources].shape[1] < max_no_sources*3: # If estimation is not max sources, pad remaining columns with nan
                
                padding = np.full((grouped_particles[no_sources].shape[0], max_no_sources*3 - grouped_particles[no_sources].shape[1]), np.nan)
                # Concatenate the original array with the padding
                grouped_particles[no_sources] = np.hstack((grouped_particles[no_sources], padding))

            no_sources_column = np.full((particle_array.shape[0], 1), no_sources)
            grouped_particles[no_sources] = np.hstack((no_sources_column, grouped_particles[no_sources]))

            particles = grouped_particles[no_sources] # Assign to separate array so that it is mutable with the for loop

            if first_append:  # First time appending
                particles_result = np.array(grouped_particles[no_sources])
                first_append = False
            else:  # For subsequent iterations
                particles_result = np.vstack((particles_result, grouped_particles[no_sources]))

        for i, particle in enumerate(particles_result):

            no_sources_change_prob = np.random.random() # Random variable from uniform distribution 0 to 1 for probability
            no_sources = int(particle[0])
                
            if 0 < no_sources_change_prob < no_sources_change_pct and no_sources < max_no_sources: # Increase number of sources belief by 1
                # Randomly initalise new source parameters
                
                particle[0] = no_sources + 1
                particle[3 * no_sources + 1] = np.random.uniform(0, search_area_x)
                particle[3 * no_sources + 2] = np.random.uniform(0, search_area_x)
                particle[3 * no_sources + 3] = np.random.uniform(min_radiation_level, max_radiation_level)
                    
            elif no_sources_change_pct < no_sources_change_prob < no_sources_change_pct*2 and no_sources > min_no_sources: # Reduce the number of sources belief by 1
                # Removing source closest to strongest particle

                # Extract number of sources (first element in the array)
                no_sources = int(particle[0])

                # Remove padding NaN values
                particle_new = particle[:no_sources*3 + 1]

                # Extract source positions and strengths
                sources = particle_new[1:].reshape(no_sources, 3)  # 3 elements per source: (x, y, strength)

                # Get the first source's (x1, y1, str1)
                x1, y1, str1 = sources[0]

                # Calculate the Euclidean distance between the first source and each of the other sources
                distances = np.linalg.norm(sources[1:, :2] - np.array([x1, y1]), axis=1)  # Only use x and y for distance

                # Find the index of the closest source
                closest_index = np.argmin(distances) + 1  # Adding 1 to skip the first source

                # Remove the closest source from the sources array
                sources = np.delete(sources, closest_index, axis=0)

                for i in range(6-no_sources):  
                    sources = np.vstack([sources, [np.nan, np.nan, np.nan]])

                # Update the particle with the remaining sources
                particle[1:] = sources.flatten()  # Flatten the array to match the original format

                particle[0] = no_sources - 1 # Subtract 1 from source count
            
            elif no_sources_change_pct*2 < no_sources_change_prob < no_sources_change_pct*2 + scramble_pct and no_sources > min_no_sources: # Can't scramble with 1 source :)
                # Scramble order of strength values

                # Extract number of sources (first element in the array)
                no_sources = int(particle[0])

                # Remove padding NaN values
                particle_new = particle[:no_sources*3 + 1]

                # Extract the 'str' values (assuming they are at every odd index)
                str_values = particle_new[3::3]

                # Ensure the shuffle is not identical to the original order
                original_str_values = str_values.copy()  # Keep a copy of the original

                # Shuffle and check if it's identical to the original
                for i in range(10): # Shuffle 10 times until it is different from original, otherwise give up
                    np.random.shuffle(str_values)
                    if not np.array_equal(str_values, original_str_values):  # If shuffled values are different from the original
                        break

                # Replace the original 'str' values with the scrambled ones
                particle_new[3::3] = str_values

                for i in range(5-no_sources):  
                    particle = np.hstack([particle_new, [np.nan, np.nan, np.nan]])

        weights_new = np.ones(N) / N
        
        return particles_result, weights_new, need_resample
    
    return particles, weights, need_resample

# Bonus: Weighted particle sample for goal-directed RL

def weighted_particle_sample(particles, weights):

    cumulative_sum = np.cumsum(weights) # Computes cumulative distribution of weights
    cumulative_sum[-1] = 1. # avoid round-off error by setting last element of cumulative_sum array to 1.0

    index = np.searchsorted(cumulative_sum, np.random.random()) # Finds index of first particle whose CDF value >= random number from 0 to 1

    return particles[index]

# Bonus 2: Sort particle source parameters within array by strength of source

def sort_sources_by_strength(particles): # Improved vectorized implementation which is 3-7x faster than original loop implementation

    # Sorts sources in descending strength order

    N = particles.shape[0]
    max_sources = (particles.shape[1] - 1) // 3
    
    # Reshape the source part into shape (N, max_sources, 3)
    sources = particles[:, 1:].reshape(N, max_sources, 3)
    
    # Get the strength values for each source (shape: (N, max_sources))
    strength = sources[:, :, 2]
    
    # Replace NaN values with -infinity so that they sort to the end in descending order
    strength_adj = np.where(np.isnan(strength), -np.inf, strength)
    
    # Get sorting indices that would sort in descending order
    # (largest strength first, and NaNs -∞ at the end)
    sort_idx = np.argsort(-strength_adj, axis=1)
    
    # Rearrange sources using these indices
    sorted_sources = np.take_along_axis(sources, sort_idx[..., None], axis=1)
    
    # Reassemble particles:
    # The first column remains the same (number of sources),
    # then the sorted sources (flattened to 1D per row).
    particles_new = np.empty_like(particles)
    particles_new[:, 0] = particles[:, 0]
    particles_new[:, 1:] = sorted_sources.reshape(N, -1)
    
    return particles_new

