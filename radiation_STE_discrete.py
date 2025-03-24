# Reinforcement learning sections of the code based on official PyTorch Reinforcement Q-Learning (DQN) tutorial 
# 
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

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
import os
import winsound

# Turns on interactive mode for MATLAB plots, so that plot is showed without use of plt.show()
plt.ion()

np.set_printoptions(threshold=25, suppress=True, precision=2) # For debugging purposes so that np.arrays print cleaner

# Variables to control display/saving of results
displayPlot = True
displaySimulation = False
displaySimulation_p = 100 # period (episodic) where simulation is displayed
savePlot = True

device = torch.device( # Checks whether CUDA/MPS device is available for acceleration, otherwise cpu is used.
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print(f"Using {device} device")

num_episodes = 2000 if torch.cuda.is_available() or torch.backends.mps.is_available() else 100

# Initialising environment

# Initialise search area/ possible source strength/ possible number of sources
search_area_x = 50 # float
search_area_y = 50 # float
min_radiation_level = 100
max_radiation_level = 300 # float
min_no_sources = 1 # This stays as an integer!
max_no_sources = 5 # This stays as an integer!

# Initalising all source parameters

sd_noise_pct = 0.10 # Sensor measurement % uncertainty

# Initalising source 1 parameters
source1_x = 35
source1_y = 40
radiation_level_1 = 200 # mSv / hour
source1 = radiation_discrete.source(source1_x, source1_y, radiation_level_1, sd_noise_pct)

# Initalising source 2 parameters
source2_x = 10
source2_y = 35
radiation_level_2 = 125 # mSv / hour
source2 = radiation_discrete.source(source2_x, source2_y, radiation_level_2, sd_noise_pct)

# Initalising source 3 parameters
source3_x = 40
source3_y = 10
radiation_level_3 = 125 # mSv / hour
source3 = radiation_discrete.source(source3_x, source3_y, radiation_level_3, sd_noise_pct)

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
rad_Z = source1.radiation_level_plot(rad_X, rad_Y) + source2.radiation_level_plot(rad_X, rad_Y) + source3.radiation_level_plot(rad_X, rad_Y) # Plot total radiation levels
Z_clipped = np.clip(rad_Z, 0, 200)

# Define the action set
actions = [agent.moveUp, agent.moveDown, agent.moveLeft, agent.moveRight]

# Initialise particle filter

N = 2500 # Number of particles

particles = pf.create_uniform_particles((0, search_area_x), (0, search_area_y), (min_radiation_level, max_radiation_level), (min_no_sources, max_no_sources), N) # Initialise particles randomly in search area/ strength/ no. of sources

particles = pf.sort_sources_by_strength(particles) # Ensure source with strongest strength comes first in the array

weights = np.ones(N) / N # All weights initalised equally

# Current (up to current episode) estimate of source parameters
largest_estimate_and_no = torch.tensor([search_area_x/2, search_area_y/2, max_radiation_level/2, max_no_sources/2]) # Initalise with guess estimate (mean of min/max bounds)

source_term_estimates = torch.zeros(0, 4, dtype=torch.float32) # Initalise tensor to track the culmulative STE to compute moving average

window_size = 25 # Moving average subset size

# Initialise a deque to hold the last 'window_size' STE values
moving_average_queue = deque(maxlen=window_size)

estimate_convergence = False # To track whether STE has converged

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
goal_dist_prob = np.array([[1/16, 1/16, 1/16, 1/16], 
                           [1/16, 1/16, 1/16, 1/16],
                           [1/16, 1/16, 1/16, 1/16],
                           [1/16, 1/16, 1/16, 1/16]]) # To keep track of the probability of going to each goal per episode

goals = np.array([[[10, 40],[20, 40],[30, 40],[40, 40]],
                  [[10, 30],[20, 30],[30, 30],[40, 30]],
                  [[10, 20],[20, 20],[30, 20],[40, 20]],
                  [[10, 10],[20, 10],[30, 10],[40, 10]]]) # To keep track of goals to visit in order to maximise knowledge about the environment

def select_goal_and_update_prob(goal_dist_prob, decay_rate=0.25):
    # Flatten the goal_dist_prob to easily pick one
    flat_prob = goal_dist_prob.flatten()
    
    # Randomly choose a goal index based on the current probability distribution
    chosen_goal_idx = np.random.choice(len(flat_prob), p=flat_prob)
    
    # Convert the flat index back to 2D coordinates (row, column)
    row, col = divmod(chosen_goal_idx, goal_dist_prob.shape[1])
    
    # Decay the probability of the chosen goal by a certain factor
    goal_dist_prob[row, col] *= (1 - decay_rate)
    
    # Normalize the probabilities so they sum to 1
    goal_dist_prob /= goal_dist_prob.sum()
    
    return (row, col), goal_dist_prob

def select_action(state, goal):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)  # Changed from (1, 16) to (1, 1)
    else:
        # Implementation of goal-directed exploration (exploratory goal method) (go towards goal half of the time)

        if np.random.random() < GOAL_PROB: # 0.5 is the probability which the agent chooses goal-directed exploration instead of a random step
            distances = np.zeros(4)

            # Find action that reduces distance between agent and goal the most

            agent_pos_array = np.array([agent.x(), agent.y()]) # Gets agent position as array
            particle_pos_array = np.array([goal[0], goal[1]]) # Gets strongest source position as array (always comes first because order is sorted)

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
fig_l_e, ax_l_e = plt.subplots()
fig_s_e, ax_s_e = plt.subplots()
fig_n_e, ax_n_e = plt.subplots()

if not displayPlot:
    plt.close(1)
    plt.close(2)
    plt.close(3)
    plt.close(4)
    plt.close(5)

def plot_distance(show_result=False, window_size=100):   
    # Convert array of data to torch tensor
    distances_t = torch.tensor(episode_end_distances, dtype=torch.float)

    ax_d.cla()  # Clear the current axis
    if show_result:
        ax_d.set_title('Result')
    else:
        ax_d.set_title('Training...')

    yticks = [0, 10, 20, 30, 40, 50, 60]

    ax_d.set_xlabel('Episode')
    ax_d.set_ylabel('Distance to source at the end of episode (m)')
    ax_l_e.set_ylim(min(yticks), max(yticks) + 5) # Ensure y-axis is fixed 
    ax_l_e.set_yticks(yticks)
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

def plot_length(show_result=False, window_size=100):
    # Convert array of data to torch tensor
    lengths_t = torch.tensor(episode_lengths, dtype=torch.float)

    ax_l.cla()  # Clear the current axis
    if show_result:
        ax_l.set_title('Result')
    else:
        ax_l.set_title('Training...')
    
    yticks = [50, 60, 70, 80, 90, 100]

    ax_l.set_xlabel('Episode')
    ax_l.set_ylabel('Length of episode (steps taken)')
    ax_l_e.set_ylim(min(yticks) - 5, max(yticks) + 5) # Ensure y-axis is fixed 
    ax_l_e.set_yticks(yticks)
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

def plot_loc_estimate(show_result=False):
    # Split data into separate tensors
    x_coord = source_term_estimates[:, 0]
    y_coord = source_term_estimates[:, 1]

    ax_l_e.cla()  # Clear the current axis

    if show_result:
        ax_l_e.set_title('Result')
    else:
        ax_l_e.set_title('Training...')

    yticks = [0, 10, 20, 30, 40, 50]

    ax_l_e.set_xlabel('Episode')
    ax_l_e.set_ylabel('Coordinates')
    ax_l_e.set_ylim(min(yticks), max(yticks) + 5) # Ensure y-axis is fixed 
    ax_l_e.set_yticks(yticks)

    # Plot lines (convergence excluded)
    ax_l_e.plot(x_coord.numpy(), label="X-coordinate", color="cyan")
    ax_l_e.plot(y_coord.numpy(), label="Y-coordinate", color="magenta")
    ax_l_e.legend()

    # Pause for dynamic plotting
    plt.pause(0.001)

    # Return the figure and axis objects for external access
    return fig_l_e, ax_l_e

def plot_strength_estimate(show_result=False):
    # Split data into separate tensors
    source_strength = source_term_estimates[:, 2]

    ax_s_e.cla()  # Clear the current axis

    if show_result:
        ax_s_e.set_title('Result')
    else:
        ax_s_e.set_title('Training...')

    yticks = [0, 50, 100, 150, 200, 250, 300]

    # Plot source strength estimate first
    ax_s_e.set_xlabel('Episode')
    ax_s_e.set_ylabel('Counts per minute (CPM)')
    ax_s_e.set_ylim(min(yticks), max(yticks) + 25) # Ensure y-axis is fixed 
    ax_s_e.set_yticks(yticks)
    ax_s_e.plot(source_strength.numpy(), label="Source strength", color="gray")

    ax_s_e.legend()

    # Pause for dynamic plotting
    plt.pause(0.001)

    # Return the figure and axis objects for external access
    return fig_s_e, ax_s_e

def plot_number_estimate(show_result=False):
    # Split data into separate tensors
    source_no = source_term_estimates[:, 3]

    ax_n_e.cla()  # Clear the current axis

    if show_result:
        ax_n_e.set_title('Result')
    else:
        ax_n_e.set_title('Training...')

    yticks = [0, 1, 2, 3, 4, 5]

    # Plot number of sources
    ax_n_e.set_xlabel('Episode')
    ax_n_e.set_ylabel('Number of sources')
    ax_n_e.set_ylim(min(yticks), max(yticks) + 1) # Ensure y-axis is fixed 
    ax_n_e.set_yticks(yticks)
    ax_n_e.plot(source_no.numpy(), label="Number of sources", color=(0.8, 0.5, 0.2))
    
    # Show legend
    ax_n_e.legend()

    # Pause for dynamic plotting
    plt.pause(0.001)

    # Return the figure and axis objects for external access
    return fig_n_e, ax_n_e

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

    # chosen_goal, goal_dist_prob = select_goal_and_update_prob(goal_dist_prob) # Choose goal for this episode and update probability distribution

    # Use estimates as goal to confirm/ fix estimate
    if i_episode == 0: # First episode
        goal = [25, 25] # Choose middle of search area as goal

    else:
        source_goal_or_random_goal = random.randint(1,2)
        if source_goal_or_random_goal == 1: # Choose random goal from array of goal coordinates
            chosen_goal, goal_dist_prob = select_goal_and_update_prob(goal_dist_prob)
            goal = goals[chosen_goal]
        elif source_goal_or_random_goal == 2: # Choose one of the sources as goal coordinates
            goal_chosen = random.randint(1, len(all_STE_x_mean))
            if goal_chosen == 1:
                goal = [largest_estimate_and_no[0], largest_estimate_and_no[1]]
            else:
                goal = [all_STE_x_mean[goal_chosen - 1], all_STE_y_mean[goal_chosen - 1]]


    for t in count():

        # Calculate total radiation level (from all sources)

        total_radiation_level = source1.radiation_level(agent.x(), agent.y()) + source2.radiation_level(agent.x(), agent.y()) + source3.radiation_level(agent.x(), agent.y())
        total_radiation_level += np.random.normal(loc=0, scale=sd_noise_pct*total_radiation_level) # Add sensor noise to total reading

        # Update/ resample particles in PF
        likelihood = pf.likelihood(agent.x(), agent.y(), particles, total_radiation_level, sd_noise_pct, min_radiation_level, max_radiation_level) # Compute likelihood of each particle

        weights = pf.update_weights(weights, likelihood) # Update weights according to likelihood

        particles, weights, need_resample, pertubations = pf.resampling_simple(particles, weights, min_no_sources, max_no_sources, min_radiation_level, max_radiation_level) # Resample if needed

        if need_resample:
            particles = pf.sort_sources_by_strength(particles) # Only need to re-sort if resampling occurs (weight changes do not affect particle parameters)

        all_STE_x_mean, all_STE_y_mean, all_STE_strength_mean, all_STE_x_var, all_STE_y_var, all_STE_strength_var,= pf.estimate(particles, weights) # Fetch tentative estimate of source location/ strength/ no. of sources

        # print(f"\nAgent location: ({agent.x()},{agent.y()})")
        # print(f"Measured: {total_radiation_level:.3f}")

        radiation_using_prediction = 0

        for j in range(len(all_STE_x_mean)):
            dist = np.sqrt((agent.x() - all_STE_x_mean[j])**2 + (agent.y() - all_STE_y_mean[j])**2)
            if dist < 1:
                radiation_using_prediction += all_STE_strength_mean[j]
            else:
                radiation_using_prediction += all_STE_strength_mean[j] / dist**2

        # print(f"Predicted radiation: {radiation_using_prediction:.3f}")

        action = select_action(state, goal)

        i = int(action.item())
        actions[i]()

        observation = agent.state()
        reward = -0.01

# Terminate on both the convergence of source estimation/ close to source term estimate

        if np.linalg.norm(np.array([largest_estimate_and_no[0], largest_estimate_and_no[1]]) - np.array([agent.x(), agent.y()])) <= 2.0 and estimate_convergence == True: # Terminate episode if agent thinks it is within 2 meters of source, and only if the STE converged already
            terminated = True
            truncated = False
            reward = 0.025*total_radiation_level
            print(f"Reward: {reward:.3f} (reached within 2 m of source estimate)")
        elif agent.actionPossible() == False:
            terminated = True
            truncated = False
            agent.moveCount = 100 # So that results plot of episode length only shows below 100 for successful episodes
            reward = -0.025*largest_estimate_and_no[2]
            print(f"Reward: {reward:.3f} (exited search area)")
        elif agent.count() >= 100:
            terminated = False
            truncated = True
            reward = 0.01*(largest_estimate_and_no[2] - 3*(math.sqrt(largest_estimate_and_no[2] / total_radiation_level) - 2))
            print(f"Reward: {reward:.3f} (episode reached 100 steps)")
        else:
            terminated = False
            truncated = False

        reward = torch.tensor([reward], device=device)

        if i_episode % displaySimulation_p == 0 and displaySimulation: # Only show simulation of episodes which are multiples of 100 (including 0).
            plt.figure(2)
            plt.clf()  # Clear the figure
    
            # Re-plot the radiation map and radiation source
            plt.contourf(rad_X, rad_Y, Z_clipped, 400, cmap='viridis')
            plt.colorbar()
            plt.plot(source1.x(), source1.y(), 'ro', markersize=4)
            plt.plot(source2.x(), source2.y(), 'o', markersize=4, color='lightcoral')
            plt.plot(source3.x(), source3.y(), 'o', markersize=4, color='lightcoral')
    
            # Append the agent's current position to the path
            agent_path.append((agent.x(), agent.y()))

            # Show the path of the agent
            for pos in agent_path:
                plt.plot(pos[0], pos[1], marker='o', color=(0.2, 0.8, 0), markersize=2)

            # Show the agent's current position as a distinct marker
            plt.plot(agent.x(), agent.y(), marker='x', color=(0.2, 0.8, 0), markersize=6)

            # Show particles in particle filter
            for particle in particles:
                for source in range(int(particle[0])): # Iterate over every source the particle is predicting                    
                    plt.plot(particle[3*source + 1], particle[3*source + 2], marker='.', color=(1, 1, 1), markersize=1)
    
            # Pause briefly to update the plot
            plt.xlim(0, search_area_x)
            plt.ylim(0, search_area_y)
            plt.pause(0.000001)

        done = terminated or truncated

        if done:
            next_state = None

            num_sources_int = particles[:,0].astype(int)

            bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5] # 6 bin edges for source no. prediction counts from 1 to 5

            source_counts, bins = np.histogram(num_sources_int, bins=bins)

            largest_STE_mean = torch.tensor([
                all_STE_x_mean[0], 
                all_STE_y_mean[0], 
                all_STE_strength_mean[0], 
                np.argmax(source_counts) + 1  # Convert to a scalar directly
            ], dtype=torch.float32)

            # Compute moving average with 'window_size' no. of values or all available values.
            moving_average_queue.append(largest_STE_mean)

            largest_estimate_and_no = torch.mean(torch.stack(list(moving_average_queue)), dim=0)


            # Determine convergence with variance of particle estimates

            # Define tolerance which checks the convergence of estimate against variance
            tolerance_loc = 4.0
            tolerance_str = 0.10 # To check the constistency of no. of sources estimation

            source_estimate_decimal = largest_estimate_and_no[3] - int(largest_estimate_and_no[3])

            if np.sqrt(all_STE_x_var[0]) < tolerance_loc and np.sqrt(all_STE_y_var[0]) < tolerance_loc and source_estimate_decimal < tolerance_str:
                estimate_convergence = True
            else:
                estimate_convergence = False
            
            print(f"Goal chosen for episode: ({goal[0]:.2f}, {goal[1]:.2f})")
            print(f"Estimated strongest source x: {largest_estimate_and_no[0]:.2f}, Estimated strongest source y: {largest_estimate_and_no[1]:.2f}, Estimated strongest source strength: {largest_estimate_and_no[2]:.2f}, Estimated number of sources: {largest_estimate_and_no[3]:.2f}")
            print(f"Strongest source x standard deviation: {np.sqrt(all_STE_x_var[0]):.2f}, Strongest source y standard deviation: {np.sqrt(all_STE_y_var[0]):.2f}, Strongest source strength standard deviation: {np.sqrt(all_STE_strength_var[0]):.2f}")
            print(f"Estimate convergence: {'True' if estimate_convergence else 'False'} \n")\
            
            small_STE_x_mean = all_STE_x_mean[1:]
            small_STE_y_mean = all_STE_y_mean[1:]
            small_STE_strength_mean = all_STE_strength_mean[1:]

            small_STE_x_var = all_STE_x_var[1:]
            small_STE_y_var = all_STE_y_var[1:]
            small_STE_strength_var = all_STE_strength_var[1:]

            for i, mean in enumerate(small_STE_x_mean):
                print(f"Estimated source {i+2} x: {small_STE_x_mean[i]:.2f}, Estimated source {i+2} y: {small_STE_y_mean[i]:.2f}, Estimated source {i+2} strength: {small_STE_strength_mean[i]:.2f}")
                print(f"Source {i+2} x standard deviation: {np.sqrt(small_STE_strength_var[i]):.2f}, Source {i+2} y standard deviation: {np.sqrt(small_STE_strength_var[i]):.2f}, Source {i+2} standard deviation: {np.sqrt(small_STE_strength_var[i]):.2f}")

            print(f"\nSource no. belief distribution --> 1: {source_counts[0]} 2: {source_counts[1]} 3: {source_counts[2]} 4: {source_counts[3]} 5: {source_counts[4]}\n\n")

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
            episode_end_distances.append(source1.distance(agent.x(), agent.y()))
            episode_lengths.append(agent.count())

            largest_estimate_and_no = largest_estimate_and_no.unsqueeze(0) # To turn current_estimate into a 2D tensor
            source_term_estimates = torch.cat((source_term_estimates, largest_estimate_and_no), dim=0)
            largest_estimate_and_no = largest_estimate_and_no.flatten(0) # To turn current_estimate back into a 1D tensor

            if displayPlot:
                fig_d, ax_d = plot_distance()
                fig_l, ax_l = plot_length()
                fig_l_e, ax_l_e = plot_loc_estimate()
                fig_s_e, ax_s_e = plot_strength_estimate()
                fig_n_e, ax_n_e  = plot_number_estimate()
            break


print('Complete')

# Indicate training is complete and show results
plot_distance(show_result=True)
plot_length(show_result=True)
plot_loc_estimate(show_result=True)
plot_strength_estimate(show_result=True)
plot_number_estimate(show_result=True)

if savePlot: # Save plot object as tuple in external file using pickle

    current_dir = os.getcwd()
    sub_dir = "multi_source_results"

    with open(os.path.join(sub_dir, "distance_plot.pkl"), "wb") as file:
        pickle.dump((fig_d, ax_d), file)

    with open(os.path.join(sub_dir,"length_plot.pkl", "wb")) as file:
        pickle.dump((fig_l, ax_l), file)

    with open(os.path.join(sub_dir,"location_plot.pkl", "wb")) as file:
        pickle.dump((fig_l_e, ax_l_e), file)
        
    with open(os.path.join(sub_dir,"strength_plot.pkl", "wb")) as file:
        pickle.dump((fig_s_e, ax_s_e), file)

    with open(os.path.join(sub_dir,"number_plot.pkl", "wb")) as file:
        pickle.dump((fig_n_e, ax_n_e), file)

    folder_path = os.path.join(current_dir, sub_dir)
    print(f"File saved at: {folder_path}")

winsound.MessageBeep() # Just to give a sound a notification that simulation is complete

plt.ioff()
plt.show()
