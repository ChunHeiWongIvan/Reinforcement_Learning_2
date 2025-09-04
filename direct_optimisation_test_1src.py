import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import radiation_discrete as rd

# Setup environment
search_area_size = 50
true_source = rd.source(10, 40, 200, 0.1)  # true location (30, 35)
agent_obj = rd.agent(5, 5, search_area_size, search_area_size, 1)

# Take N measurements
obs_positions = []
obs_measurements = []
for i in range(10): # Number of observations made
    rand_dir = np.random.randint(1, 5)
    if rand_dir == 1:
        agent_obj.moveUp()
    elif rand_dir == 2:
        agent_obj.moveRight()
    elif rand_dir == 3:
        agent_obj.moveDown()
    elif rand_dir == 4:
        agent_obj.moveLeft()

    true_level = true_source.radiation_level(agent_obj.x(), agent_obj.y())
    noise = np.random.normal(0, true_source.sd_noise_pct * true_level)
    observed = true_level + noise
    obs_positions.append((agent_obj.x(), agent_obj.y()))
    obs_measurements.append(observed)

# Define error function to minimize (source_x, source_y, strength)
def error_fn(params):
    sx, sy, strength = params
    error = 0
    for (x, y), observed in zip(obs_positions, obs_measurements):
        dist_sq = (x - sx)**2 + (y - sy)**2
        predicted = strength / dist_sq if dist_sq != 0 else strength
        error += (predicted - observed)**2
    return error

# Initial guess: center and medium strength
initial_guess = [25, 25, 150]
result = minimize(error_fn, initial_guess, bounds=[(0, 50), (0, 50), (50, 500)])
estimated_x, estimated_y, estimated_strength = result.x

# Prepare plot and terminal outputs
fig, ax = plt.subplots(figsize=(6, 6))
obs_xs, obs_ys = zip(*obs_positions)
ax.scatter(obs_xs, obs_ys, c='blue', label='Observations')
ax.scatter([true_source.source_x], [true_source.source_y], c='green', marker='*', s=200, label='True Source')
ax.scatter([estimated_x], [estimated_y], c='red', marker='x', s=100, label='Estimated Source')
ax.set_xlim(0, 50)
ax.set_ylim(0, 50)
ax.set_title('Source Localisation using Optimisation')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.grid(True)
ax.legend()

# Print outputs
terminal_output = {
    "observations": list(zip(obs_positions, np.round(obs_measurements, 2))),
    "estimated": {
        "x": round(estimated_x, 2),
        "y": round(estimated_y, 2),
        "strength": round(estimated_strength, 2)
    },
    "true": {
        "x": true_source.source_x,
        "y": true_source.source_y,
        "strength": true_source.source_radioactivity
    }
}

print("\nObservations (Position, Measured Level):")
for pos, meas in terminal_output["observations"]:
    print(f"  {pos} -> {meas}")

print("\nEstimated Source Location:")
print(f"  x: {terminal_output['estimated']['x']}")
print(f"  y: {terminal_output['estimated']['y']}")
print(f"  strength: {terminal_output['estimated']['strength']}")

print("\nTrue Source Location:")
print(f"  x: {terminal_output['true']['x']}")
print(f"  y: {terminal_output['true']['y']}")
print(f"  strength: {terminal_output['true']['strength']}")

plt.show()