import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import radiation_discrete
import particle_filter as pf
import pickle
import winsound

# Turns on interactive mode for MATLAB plots, so that plot is showed without use of plt.show()
plt.ion()

# Variables to control display/saving of results
displayPlot = True
displaySimulation = False
displaySimulation_f = 100 # frequency (episodic) where simulation is displayed
savePlot = True

device = torch.device( # Checks whether CUDA/MPS device is available for acceleration, otherwise cpu is used.
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print(f"Using {device} device")

num_episodes = 3000 if torch.cuda.is_available() or torch.backends.mps.is_available() else 100

# Initialising environment

# Initialise search area/ possible source strength
search_area_x = 50
search_area_y = 50
max_radiation_level = 500

# Initalising source parameters
source_x = 25
source_y = 40
radiation_level = 100 # mSv / hour
sd_noise_pct = 0.10 # Sensor measurement % uncertainty
source = radiation_discrete.source(source_x, source_y, radiation_level, sd_noise_pct)

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
rad_Z = source.radiation_level_plot(rad_X, rad_Y)
Z_clipped = np.clip(rad_Z, 0, 100)

# Define the action set
actions = [agent.moveUp, agent.moveDown, agent.moveLeft, agent.moveRight]

# Initialise particle filter

N = 1000 # Number of particles

particles = pf.create_uniform_particles((0, search_area_x), (0, search_area_y), (0, max_radiation_level), N) # Initialise particles randomly in search area

weights = np.ones(N) / N # All weights initalised equally

# Current (up to current episode) estimate of source parameters
current_estimate = torch.tensor([search_area_x/2, search_area_y/2, max_radiation_level/2]) # Initalise with guess estimate (mean of min/max bounds)

source_term_estimates = torch.zeros(0, 3, dtype=torch.float32) # Initalise tensor to track the culmulative STE to compute moving average
# current_estimate = current_estimate.unsqueeze(0) # To turn current_estimate into a 2D tensor
# source_term_estimates = torch.cat((source_term_estimates, current_estimate), dim=0)
# current_estimate = current_estimate.flatten(0) # To turn current_estimate back into a 1D tensor

window_size = 50 # Moving average subset size

# Initialise a deque to hold the last 'window_size' STE values
moving_average_queue = deque(maxlen=window_size)

estimate_convergence = False # To track whether STE has converged
episode_converged = None # To track at which episode the estimate first converges

# Named tuple allows access to its elements using named fields. Unlike dictionaries, more memory efficient and the named fields are immutable.
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    # deque = double-ended queue, can contain any data type, O(1) time complexity for adding/removing from both ends.
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) 

    # Updates memory with a new experience (contained in the 'Transition' named tuple). Each element in memory stores a named tuple.
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    # Randomly samples a batch of experiences into a list of named tuples.
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # Returns the current number of experiences in memory.
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 2500)
        self.layer2 = nn.Linear(2500, 2500)
        self.layer3 = nn.Linear(2500, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.00
EPS_DECAY = 50000 # Episilon decays per step done (20000 steps ~ 200 full length episodes)
TAU = 0.005
LR = 0.002 # was 0.001
GOAL_PROB = 0.5 # Probability of goal-directed exploration instead of random exploration

n_actions = len(actions)
n_observations = (search_area_x + 1) * (search_area_y + 1)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state, particle_set, weight_set):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)  # Changed from (1, 16) to (1, 1)
    else:
        # Implementation of goal-based exploration (exploratory goal method) (go towards goal half of the time)

        if np.random.random() < GOAL_PROB: # 0.5 is the probability which the agent chooses goal-based exploration instead of a random step
            particle = pf.weighted_particle_sample(particle_set, weight_set) # Sample weighted random particle to use as goal
            distances = np.zeros(4)

            # Find action that reduces distance between agent and goal the most

            agent_pos_array = np.array([agent.x(), agent.y()]) # Gets agent position as array
            particle_pos_array = np.array([particle[0], particle[1]]) # Gets particle position as array

            # Compute distance after moveUp
            distances[0] = np.linalg.norm(agent_pos_array + np.array([0.0, 1.0]) - particle_pos_array) # adding (0,1) corresponds to moving up
            # Compute distance after moveDown
            distances[1] = np.linalg.norm(agent_pos_array + np.array([0.0, -1.0]) - particle_pos_array) # adding (0,-1) corresponds to moving down
            # Compute distance after moveLeft
            distances[2] = np.linalg.norm(agent_pos_array + np.array([-1.0, 0.0]) - particle_pos_array) # adding (-1,0) corresponds to moving left
            # Compute distance after moveRight
            distances[3] = np.linalg.norm(agent_pos_array + np.array([1.0, 0.0]) - particle_pos_array) # adding (1,0) corresponds to moving right

            return torch.tensor([[np.argmin(distances)]]).to(device) # navigate towards random particle half of the time

        else:
            return torch.tensor([[random.randint(0,3)]], device=device, dtype=torch.long) # for selecting random action


episode_end_distances = [] # Track distance to source at end of episode
episode_lengths = [] # Track lengths of episodes

# Initialise plot objects
fig_d, ax_d = plt.subplots()
fig_l, ax_l = plt.subplots()
fig_e, ax_e = plt.subplots()

plot_x = [] # To track episodes where estimate has converged to use as x-coordinates

def plot_distance(fig_d, ax_d, show_result=False, window_size=100):   
    # Convert array of data to torch tensor
    distances_t = torch.tensor(episode_end_distances, dtype=torch.float)

    ax_d.cla()  # Clear the current axis
    if show_result:
        ax_d.set_title('Result')
    else:
        ax_d.set_title('Training...')

    ax_d.set_xlabel('Episode')
    ax_d.set_ylabel('Distance to source at the end of episode')
    ax_d.plot(distances_t.numpy(), label="Raw data")
    
    # Compute average values
    if len(distances_t) < window_size:
        # Compute running average for all episodes if < window_size
        running_average = torch.cumsum(distances_t, dim=0) / torch.arange(1, len(distances_t) + 1)
        ax_d.plot(running_average.numpy(), label='Moving average')
    else:
        # Combine running average (for episodes < window_size) and moving average
        running_average = torch.cumsum(distances_t[:window_size - 1], dim=0) / torch.arange(1, window_size)
        means_after_window = distances_t.unfold(0, window_size, 1).mean(1)
        full_average = torch.cat((running_average, means_after_window))
        ax_d.plot(full_average.numpy(), label='Moving average')

    ax_d.legend()

    # Pause for dynamic plotting
    plt.pause(0.001)
    
    # Return the figure and axis objects for external access
    return fig_d, ax_d

def plot_length(fig_l, ax_l, show_result=False, window_size=100):
    # Convert array of data to torch tensor
    lengths_t = torch.tensor(episode_lengths, dtype=torch.float)

    ax_l.cla()  # Clear the current axis
    if show_result:
        ax_l.set_title('Result')
    else:
        ax_l.set_title('Training...')

    ax_l.set_xlabel('Episode')
    ax_l.set_ylabel('Length of episode')
    ax_l.plot(lengths_t.numpy(), label="Raw data")
    
    # Compute average values
    if len(lengths_t) < window_size:
        # Compute running average for all episodes if < window_size
        running_average = torch.cumsum(lengths_t, dim=0) / torch.arange(1, len(lengths_t) + 1)
        ax_l.plot(running_average.numpy(), label='Moving average')
    else:
        # Combine running average (for episodes < window_size) and moving average
        running_average = torch.cumsum(lengths_t[:window_size - 1], dim=0) / torch.arange(1, window_size)
        means_after_window = lengths_t.unfold(0, window_size, 1).mean(1)
        full_average = torch.cat((running_average, means_after_window))
        ax_l.plot(full_average.numpy(), label='Moving average')

    ax_l.legend()

    # Pause for dynamic plotting
    plt.pause(0.001)
    
    # Return the figure and axis objects for external access
    return fig_l, ax_l

def plot_estimate(fig_e, ax_e, show_result=False):
    # Split data into separate tensors
    x_coord = source_term_estimates[:, 0]
    y_coord = source_term_estimates[:, 1]
    source_strength = source_term_estimates[:, 2]

    ax_e.cla()  # Clear the current axis

    if show_result:
        ax_e.set_title('Result')
    else:
        ax_e.set_title('Training...')

    ax_e.set_xlabel('Episode')
    ax_e.set_ylabel('Source term estimate')

    # Plot the lines when the estimate has converged

    if not estimate_convergence:
        ax_e.plot(x_coord.numpy(), label="X-coordinate (before convergence)", color="cyan")
        ax_e.plot(y_coord.numpy(), label="Y-coordinate (before convergence)", color="magenta")
        ax_e.plot(source_strength.numpy(), label="Source strength (before convergence)", color="gray")
    else:
        # Slice tensors to keep displaying data before convergence
        x_coord_not_converged = x_coord[:episode_converged + 1]
        y_coord_not_converged = y_coord[:episode_converged + 1]
        source_strength_not_converged = source_strength[:episode_converged + 1]

        # Slice tensors to display data after convergence
        x_coord_converged = x_coord[episode_converged:]
        y_coord_converged = y_coord[episode_converged:]
        source_strength_converged = source_strength[episode_converged:]

        ax_e.plot(x_coord_not_converged.numpy(), label="X-coordinate (before convergence)", color="cyan")
        ax_e.plot(y_coord_not_converged.numpy(), label="Y-coordinate (before convergence)", color="magenta")
        ax_e.plot(source_strength_not_converged.numpy(), label="Source strength (before convergence)", color="gray")

        ax_e.plot(plot_x, x_coord_converged.numpy(), label="X-coordinate (after convergence)", color="blue")
        ax_e.plot(plot_x, y_coord_converged.numpy(), label="Y-coordinate (after convergence)", color="red")
        ax_e.plot(plot_x, source_strength_converged.numpy(), label="Source strength (after convergence)", color="black")

    ax_e.legend()

    # Pause for dynamic plotting
    plt.pause(0.001)

    # Return the figure and axis objects for external access
    return fig_e, ax_e


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def one_hot_encode(state, n_observations):
    one_hot = torch.zeros(n_observations, dtype=torch.float32, device=device)
    one_hot[state] = 1
    return one_hot

for i_episode in range(num_episodes):
    agent.reset()
    state = agent.state()
    state = one_hot_encode(state, n_observations).unsqueeze(0)

    agent_path = [] # To store agent past positions to display path taken

    for t in count():
        action = select_action(state, particles, weights)
        i = int(action.item())
        actions[i]()

        observation = agent.state()
        reward = -1.0
        # reward = 0.01 * source.radiation_level(agent.x(), agent.y())

        # Update/ resample particles in PF
        likelihood = pf.likelihood(agent.x(), agent.y(), particles, source.radiation_level(agent.x(), agent.y()), sd_noise_pct) # Compute likelihood of each particle

        weights = pf.update_weights(weights, likelihood) # Update weights according to likelihood

        particles, weights = pf.resampling(particles, weights) # Resample if needed

        STE_mean, STE_var = pf.estimate(particles, weights) # Fetch tentative estimate of source location/ strength


# Terminate on both the convergence of source estimation/ close to source term estimate

        if source.radiation_level(agent.x(), agent.y()) >= current_estimate[2] and estimate_convergence == True: # Terminate episode if agent thinks it is within 1 meter of source, and only if the STE converged already
            terminated = True
            truncated = False
            reward = 2.5*current_estimate[2]
            print(f"Reward: {reward:.2f} (reached within 1 m of source estimate)")
        elif agent.actionPossible() == False:
            terminated = True
            truncated = False
            agent.moveCount = 100 # So that results plot of episode length only shows below 100 for successful episodes
            reward = -2.5*current_estimate[2]
            print(f"Reward: {reward:.2f} (exited search area)")
        elif agent.count() >= 100:
            terminated = False
            truncated = True
            reward = current_estimate[2] - 3*(math.sqrt(current_estimate[2] / source.radiation_level(agent.x(), agent.y())) - 1)
            print(f"Reward: {reward:.2f} (episode reached 100 steps)")
        else:
            terminated = False
            truncated = False

        reward = torch.tensor([reward], device=device)

        if i_episode % displaySimulation_f == 0 and displaySimulation: # Only show simulation of episodes which are multiples of 100 (including 0).
            plt.figure(2)
            plt.clf()  # Clear the figure
    
            # Re-plot the radiation map and radiation source
            plt.contourf(rad_X, rad_Y, Z_clipped, 200, cmap='viridis')
            plt.colorbar()
            plt.plot(source.x(), source.y(), 'ro', markersize=4)
    
            # Append the agent's current position to the path
            agent_path.append((agent.x(), agent.y()))

            # Show the path of the agent
            for pos in agent_path:
                plt.plot(pos[0], pos[1], marker='o', color=(0.2, 0.8, 0), markersize=2)

            # Show the agent's current position as a distinct marker
            plt.plot(agent.x(), agent.y(), marker='x', color=(0.2, 0.8, 0), markersize=6)

            # Show particles in particle filter
            for particle in particles:    
                plt.plot(particle[0], particle[1], marker='.', color=(1, 1, 1), markersize=1)
    
            # Pause briefly to update the plot
            plt.xlim(0, search_area_x)
            plt.ylim(0, search_area_y)
            plt.pause(0.00001)

        done = terminated or truncated

        if done:
            next_state = None
            moving_average_queue.append(STE_mean)

            # Compute moving average with 'window_size' no. of values or all available values.

            old_estimate = current_estimate
            current_estimate = torch.mean(torch.stack(list(moving_average_queue)), dim=0)

            # Define tolerance which checks the convergence of moving average
            tolerance_loc = 0.025
            tolerance_rad = 0.125

            difference = current_estimate - old_estimate # Compute difference in moving averages

            loc_array = np.array([difference[0], difference[1]])
            rad = difference[2]

            if np.all(np.abs(loc_array) < tolerance_loc) and np.abs(rad) < tolerance_rad:
                estimate_convergence = True
                if episode_converged == None: # Keep tracks of no. of episode where convergence first occurs
                    episode_converged = i_episode
                
            
            if not episode_converged == None:
                plot_x.append(i_episode) # Keep an array of all episodes since convergence for plotting


            print(f"Estimated source x: {current_estimate[0]:.2f}, Estimated source y: {current_estimate[1]:.2f}, Estimated source strength: {current_estimate[2]:.2f}")
            print(f"Estimate convergence: {'True' if estimate_convergence else 'False'} \n")

        else:
            next_state = one_hot_encode(observation, n_observations).unsqueeze(0)

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        # Update target network
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_end_distances.append(source.distance(agent.x(), agent.y()))
            episode_lengths.append(agent.count())

            current_estimate = current_estimate.unsqueeze(0) # To turn current_estimate into a 2D tensor
            source_term_estimates = torch.cat((source_term_estimates, current_estimate), dim=0)
            current_estimate = current_estimate.flatten(0) # To turn current_estimate back into a 1D tensor

            if displayPlot:
                fig_d, ax_d = plot_distance(fig_d, ax_d)
                fig_l, ax_l = plot_length(fig_l, ax_l)
                fig_e, ax_e = plot_estimate(fig_e, ax_e)
            break


print('Complete')

# Indicate training is complete and show results
plot_distance(fig_d, ax_d, show_result=True)
plot_length(fig_l, ax_l, show_result=True)
plot_estimate(fig_e, ax_e, show_result=True)

if savePlot: # Save plot object as tuple in external file using pickle
    with open("distance_plot_goal.pkl", "wb") as file:
        pickle.dump((fig_d, ax_d), file)

    with open("length_plot_goal.pkl", "wb") as file:
        pickle.dump((fig_l, ax_l), file)

    with open("estimate_plot_goal.pkl", "wb") as file:
        pickle.dump((fig_e, ax_e), file)

winsound.MessageBeep() # Just to give notification that simulation is complete

plt.ioff()
plt.show()
