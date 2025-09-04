import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import radiation_discrete as rd

# ========= PARAMETERS =========
no_sources = 3                         # <<< CHANGE THIS TO ANY NUMBER OF SOURCES
search_area_size = 50
agent_start = (5, 5)
n_observations = 100
noise_pct = 0.1
min_distance = 5  # minimum allowed distance between sources
# ==============================

# ----- Create true sources ensuring separation -----
true_sources = []

while len(true_sources) < no_sources:
    sx = np.random.randint(10, 40)
    sy = np.random.randint(10, 40)
    strength = np.random.uniform(150, 250)
    new_source = rd.source(sx, sy, strength, noise_pct)

    # Enforce min distance from all existing sources
    too_close = any(np.hypot(sx - s.x(), sy - s.y()) < min_distance for s in true_sources)
    if not too_close:
        true_sources.append(new_source)

# ----- Create agent -----
agent_obj = rd.agent(agent_start[0], agent_start[1], search_area_size, search_area_size, 1)

# ----- Collect observations -----
obs_positions = []
obs_measurements = []

for _ in range(n_observations):
    rand_dir = np.random.randint(1, 5)
    if rand_dir == 1:
        agent_obj.moveUp()
    elif rand_dir == 2:
        agent_obj.moveRight()
    elif rand_dir == 3:
        agent_obj.moveDown()
    elif rand_dir == 4:
        agent_obj.moveLeft()

    pos = (agent_obj.x(), agent_obj.y())
    total_true_level = sum([s.radiation_level(*pos) for s in true_sources])
    noise = np.random.normal(0, noise_pct * total_true_level)
    obs_positions.append(pos)
    obs_measurements.append(total_true_level + noise)

# ----- Define error function with separation penalty -----
def error_fn(params):
    error = 0

    # Add large penalty if sources are closer than min_distance
    for i in range(no_sources):
        xi = params[i * 3]
        yi = params[i * 3 + 1]
        for j in range(i + 1, no_sources):
            xj = params[j * 3]
            yj = params[j * 3 + 1]
            dist = np.hypot(xi - xj, yi - yj)
            if dist < min_distance:
                error += 1e6 * (min_distance - dist)**2  # big penalty

    # Sum squared error for all observations
    for (x, y), observed in zip(obs_positions, obs_measurements):
        predicted = 0
        for i in range(no_sources):
            sx = params[i * 3]
            sy = params[i * 3 + 1]
            strength = params[i * 3 + 2]
            dist_sq = (x - sx) ** 2 + (y - sy) ** 2
            predicted += strength / dist_sq if dist_sq > 1e-3 else strength  # prevent div by 0
        error += (predicted - observed) ** 2

    return error

# ----- Initial guess and bounds -----
initial_guess = []
bounds = []
for i in range(no_sources):
    initial_guess += [np.random.uniform(0, 50), np.random.uniform(0, 50), 150]
    bounds += [(0, search_area_size), (0, search_area_size), (50, 500)]

# ----- Optimisation -----
result = minimize(error_fn, initial_guess, bounds=bounds)
estimated_params = result.x

# ----- Plotting -----
fig, ax = plt.subplots(figsize=(6, 6))
obs_xs, obs_ys = zip(*obs_positions)
ax.scatter(obs_xs, obs_ys, c='blue', s=15, label='Observations')

# Plot true sources
for i, s in enumerate(true_sources):
    ax.scatter(s.x(), s.y(), marker='*', c='green', s=150, label=f'True Source #{i+1}')

# Plot estimated sources
for i in range(no_sources):
    est_x = estimated_params[i * 3]
    est_y = estimated_params[i * 3 + 1]
    ax.scatter(est_x, est_y, marker='x', c='red', s=100, label=f'Estimated Source #{i+1}')

# Final plot tweaks
ax.set_xlim(0, search_area_size)
ax.set_ylim(0, search_area_size)
ax.set_title(f'{no_sources}-Source Localisation via Optimisation')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.grid(True)
ax.legend(loc='upper right')
plt.show()

# ----- Terminal output -----
print("\nEstimated Sources:")
for i in range(no_sources):
    print(f"  #{i+1}: x={round(estimated_params[i*3],2)}, y={round(estimated_params[i*3+1],2)}, strength={round(estimated_params[i*3+2],2)}")

print("\nTrue Sources:")
for i, s in enumerate(true_sources):
    print(f"  #{i+1}: x={s.x()}, y={s.y()}, strength={s.source_radioactivity}")
