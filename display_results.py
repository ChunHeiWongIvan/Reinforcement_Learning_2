import matplotlib.pyplot as plt
import pickle
import torch

# Load the distance and length plot from pickle files
with open('distance_plot_random.pkl', 'rb') as f:  # Load distance plot (random exploration RL)
    distance_plot_random = pickle.load(f)

with open('distance_plot_goal.pkl', 'rb') as f:  # Load distance plot (goal-directed RL)
    distance_plot_goal = pickle.load(f)

with open('length_plot_random.pkl', 'rb') as f:  # Load length plot (random exploration RL)
    length_plot_random = pickle.load(f)

with open('length_plot_goal.pkl', 'rb') as f:  # Load length plot (goal-directed RL)
    length_plot_goal = pickle.load(f)

with open('estimate_plot_random.pkl', 'rb') as f:  # Load length plot (random exploration RL)
    estimate_plot_random = pickle.load(f)

with open('estimate_plot_goal.pkl', 'rb') as f:  # Load length plot (goal-directed RL)
    estimate_plot_goal = pickle.load(f)

# Extract the axes from the loaded plots
ax_d_random_loaded = distance_plot_random[1]
ax_d_goal_loaded = distance_plot_goal[1]
ax_l_random_loaded = length_plot_random[1]
ax_l_goal_loaded = length_plot_goal[1]
ax_e_random_loaded = estimate_plot_random[1]
ax_e_goal_loaded = estimate_plot_goal[1]

# Get the all lines from the distance plot
lines_d_random = ax_d_random_loaded.get_lines()
lines_d_goal = ax_d_goal_loaded.get_lines()
lines_l_random = ax_l_random_loaded.get_lines()
lines_l_goal = ax_l_goal_loaded.get_lines()
lines_e_random = ax_e_random_loaded.get_lines()
lines_e_goal = ax_e_goal_loaded.get_lines()

# Get specific line (data series array) from line object
distances_random = lines_d_random[1].get_ydata()
distances_goal = lines_d_goal[1].get_ydata()
lengths_random = lines_l_random[1].get_ydata()
lengths_goal = lines_l_goal[1].get_ydata()

estimates_random_x_nc = lines_e_random[0].get_xdata(), lines_e_random[0].get_ydata()
estimates_random_y_nc = lines_e_random[1].get_xdata(), lines_e_random[1].get_ydata()
estimates_random_src_nc = lines_e_random[2].get_xdata(), lines_e_random[2].get_ydata()
estimates_random_x_c = lines_e_random[3].get_xdata(), lines_e_random[3].get_ydata()
estimates_random_y_c = lines_e_random[4].get_xdata(), lines_e_random[4].get_ydata()
estimates_random_src_c = lines_e_random[5].get_xdata(), lines_e_random[5].get_ydata()

estimates_goal_x_nc = lines_e_goal[0].get_xdata(), lines_e_goal[0].get_ydata()
estimates_goal_y_nc = lines_e_goal[1].get_xdata(), lines_e_goal[1].get_ydata()
estimates_goal_src_nc = lines_e_goal[2].get_xdata(), lines_e_goal[2].get_ydata()
estimates_goal_x_c = lines_e_goal[3].get_xdata(), lines_e_goal[3].get_ydata()
estimates_goal_y_c = lines_e_goal[4].get_xdata(), lines_e_goal[4].get_ydata()
estimates_goal_src_c = lines_e_goal[5].get_xdata(), lines_e_goal[5].get_ydata()

# Initialize the figure and axes only for the distance plot (fig_d)
fig_d, ax_d = plt.subplots()
fig_l, ax_l = plt.subplots()
fig_e, ax_e = plt.subplots()

# Function to plot the combined distances
def plot_combined_distances(ax_d):
    # Convert data arrays into tensors
    distances_random_tensor = torch.tensor(distances_random, dtype=torch.float)
    distances_goal_tensor = torch.tensor(distances_goal, dtype=torch.float)

    ax_d.set_xlabel('Episodes')
    ax_d.set_ylabel('Average distance to source at end of episode')

    # Plot the data (1st series)
    ax_d.plot(distances_random_tensor.numpy(), label="Random exploration RL", color='blue', linestyle='--')

    # Plot the data (2nd series)
    ax_d.plot(distances_goal_tensor.numpy(), label="Goal-directed RL", color='blue', linestyle='-')

    ax_d.legend()

def plot_combined_lengths(ax_l):
    # Convert data arrays into tensors
    lengths_random_tensor = torch.tensor(lengths_random, dtype=torch.float)
    lengths_goal_tensor = torch.tensor(lengths_goal, dtype=torch.float)

    ax_l.set_xlabel('Episodes')
    ax_l.set_ylabel('Average episode length')

    # Plot the data (1st series)
    ax_l.plot(lengths_random_tensor.numpy(), label="Random exploration RL", color='blue', linestyle='--')

    # Plot the data (2nd series)
    ax_l.plot(lengths_goal_tensor.numpy(), label="Goal-directed RL", color='blue', linestyle='-')

    ax_l.legend()

def plot_combined_estimates(ax_e):

    ax_l.set_xlabel('Episodes')
    ax_l.set_ylabel('Source estimate parameters')

    # Plot the data (1st series for random exploration)
    ax_e.plot(estimates_random_x_nc[0], estimates_random_x_nc[1], color='#ADD8E6', linestyle='--')
    ax_e.plot(estimates_random_y_nc[0], estimates_random_y_nc[1], color='#FFCCCB', linestyle='--')
    ax_e.plot(estimates_random_src_nc[0], estimates_random_src_nc[1], color='gray', linestyle='--')
    ax_e.plot(estimates_random_x_c[0], estimates_random_x_c[1], label="Random exploration RL (x-coordinate)", color='blue', linestyle='--')
    ax_e.plot(estimates_random_y_c[0], estimates_random_y_c[1], label="Random exploration RL (y-coordinate)", color='red', linestyle='--')
    ax_e.plot(estimates_random_src_c[0], estimates_random_src_c[1], label="Random exploration RL (source strength)", color='black', linestyle='--')

    # Plot the data (2nd series for goal-directed)
    ax_e.plot(estimates_goal_x_nc[0], estimates_goal_x_nc[1], color='#ADD8E6', linestyle='-')
    ax_e.plot(estimates_goal_y_nc[0], estimates_goal_y_nc[1], color='#FFCCCB', linestyle='-')
    ax_e.plot(estimates_goal_src_nc[0], estimates_goal_src_nc[1], color='gray', linestyle='-')
    ax_e.plot(estimates_goal_x_c[0], estimates_goal_x_c[1], label="Goal-directed RL (x-coordinate)", color='blue', linestyle='-')
    ax_e.plot(estimates_goal_y_c[0], estimates_goal_y_c[1], label="Goal-directed RL (y-coordinate)", color='red', linestyle='-')
    ax_e.plot(estimates_goal_src_c[0], estimates_goal_src_c[1], label="Goal-directed RL (source strength)", color='black', linestyle='-')

    ax_e.legend(fontsize=9)

# Call the functions to plot combined data for distances/ lengths
plot_combined_distances(ax_d)
plot_combined_lengths(ax_l)
plot_combined_estimates(ax_e)

for i in range(6): # 6 of the original plots have to be closed
    plt.close(i+1)

plt.show()
