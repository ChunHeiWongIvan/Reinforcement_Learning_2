# Reinforcement learning sections of the code based on official PyTorch Reinforcement Q-Learning (DQN) tutorial
#
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW # for calaculating weighted variance
import random
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.collections as collections
from collections import namedtuple, deque
from itertools import count
import radiation_discrete
import particle_filter as pf
import pickle
import os

# Turns on interactive mode for MATLAB plots, so that plot is showed without use of plt.show()
plt.ion()

# Variables to control display/saving of results
displayPlot = True
displaySimulation = True
displaySimulation_p = 100 # period (episodic) where simulation is displayed
savePlot = True

device = torch.device( # Checks whether CUDA/MPS device is available for acceleration, otherwise cpu is used.
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print(f"Using {device} device")

num_episodes = 2000 if torch.cuda.is_available() or torch.backends.mps.is_available() else 500

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
radiation_level_2 = 140 # mSv / hour
source2 = radiation_discrete.source(source2_x, source2_y, radiation_level_2, sd_noise_pct)

# Initalising source 3 parameters
source3_x = 40
source3_y = 10
radiation_level_3 = 120 # mSv / hour
source3 = radiation_discrete.source(source3_x, source3_y, radiation_level_3, sd_noise_pct)

# Initialising agent parameters
agent_x = 12
agent_y = 12
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

# particles = pf.create_cheat_particles(3, source1, source2, source3, (min_no_sources, max_no_sources), N)

particles = pf.sort_sources_by_strength(particles) # Ensure source with strongest strength comes first in the array

weights = np.ones(N) / N # All weights initalised equally

mean_no_sources = (min_no_sources + max_no_sources) / 2

# Track mean of belief state (moving average)
belief_state_mean = torch.cat([torch.tensor([mean_no_sources]), 
                          torch.tensor([search_area_x/2, search_area_y/2, (min_radiation_level + max_radiation_level) / 2]).repeat(int(mean_no_sources)),
                          torch.tensor([np.nan, np.nan, np.nan]).repeat(max_no_sources - int(mean_no_sources))]) # Initalise with guess estimate of all parameters (mean of min/max bounds), pad empty sources with np.nan

# Track variance of belief state (moving average)
belief_state_var = torch.cat([torch.tensor([mean_no_sources]), 
                          torch.tensor([0.25, 0.25, 0.25]).repeat(int(mean_no_sources)), # 0.25 is 0.5^2, which is the initial perturbance used in resampling
                          torch.tensor([np.nan, np.nan, np.nan]).repeat(max_no_sources - int(mean_no_sources))]) # Initalise with guess estimate of all parameters (mean of min/max bounds), pad empty sources with np.nan

belief_state_mean_over_time = torch.zeros(0, 3*max_no_sources + 1, dtype=torch.float32) # Initalise tensor to track the culmulative STE to compute moving average

window_size = 25 # Moving average subset size

# Initialise a deque to hold the last 'window_size' STE mean to compute moving average
moving_average_queue_mean = deque(maxlen=window_size)

# Initalise a deque to hold the last 'window_size' STE var to compute moving average
moving_average_queue_var = deque(maxlen=window_size)

convergence_stable = np.array([False, False, False]) # To track whether STEs is stable and converged
episode_converged = np.zeros(3, dtype=np.int64) # To track at which episode STEs is converged and stable
convergence_tracking = np.zeros(3, dtype=np.int64) # Require a sustained period of convergence before actually recognising stability (tracking all sources)
plot_x_1 = [] # To keep track of all episodes since convergence
plot_x_2 = [] # To keep track of all episodes since convergence
plot_x_3 = [] # To keep track of all episodes since convergence

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
EPS_END = 0.0
EPS_DECAY = 80000 # Episilon decays per step done (changed from 80k for non-goal)
TAU = 0.005
LR = 0.001 # Sparse rewards so lower LR might help stabilise training
GOAL_PROB = 0.5 # Probability of goal-directed exploration instead of random exploration

n_actions = len(actions)
n_observations = (search_area_x + 1) * (search_area_y + 1)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000) # 10000 = 100 full length episodes

steps_done = 0 # Track number of steps done overall
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
map_plotted = False # Flag for whether radiation map has been plotted
legend_plotted = False # Flag for whether legend has been plotted

# Initialise plot objects
fig_sim, ax_sim = plt.subplots(figsize=(9.6, 7.2))
fig_d, ax_d = plt.subplots()
fig_l, ax_l = plt.subplots()
fig_l_e, ax_l_e = plt.subplots()
fig_s_e, ax_s_e = plt.subplots()
fig_n_e, ax_n_e = plt.subplots()
fig_l_er, ax_l_er = plt.subplots()

if not displaySimulation:
    plt.close(1)

if not displayPlot:
    plt.close(2)
    plt.close(3)
    plt.close(4)
    plt.close(5)
    plt.close(6)
    plt.close(7)

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

    yticks = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    # yticks = [50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290] # For 300 step length for no-goal based

    ax_l.set_xlabel('Episode')
    ax_l.set_ylabel('Length of episode (steps taken)')
    ax_l.set_ylim(min(yticks) - 5, max(yticks) + 15) # Ensure y-axis is fixed
    ax_l.set_yticks(yticks)
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
    x_coord = belief_state_mean_over_time[:, 1]
    y_coord = belief_state_mean_over_time[:, 2]

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

def plot_strength_estimate_error(show_result=False):
    
    # Split data into separate tensors
    source_strength_1_error = abs(belief_state_mean_over_time[:, 3] - source1.source_radioactivity)
    source_strength_2_error = abs(belief_state_mean_over_time[:, 6] - source2.source_radioactivity)
    source_strength_3_error = abs(belief_state_mean_over_time[:, 9] - source3.source_radioactivity)

    ax_s_e.cla()  # Clear the current axis

    if show_result:
        ax_s_e.set_title('Result')
    else:
        ax_s_e.set_title('Training...')

    # Plot source strength estimate first
    ax_s_e.set_xlabel('Episode')
    ax_s_e.set_ylabel('Equivalent dose rate (mSv/h)')
    ax_s_e.plot(source_strength_1_error.numpy(), label="Source 1", color="black")
    ax_s_e.plot(source_strength_2_error.numpy(), label="Source 2", color="darkgray")
    ax_s_e.plot(source_strength_3_error.numpy(), label="Source 3", color="lightgray")

    ax_s_e.legend()

    # Pause for dynamic plotting
    plt.pause(0.001)

    # Return the figure and axis objects for external access
    return fig_s_e, ax_s_e

def plot_number_estimate(show_result=False):
    # Split data into separate tensors
    source_no = belief_state_mean_over_time[:, 0]

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

def plot_loc_error(show_result=False): # Not automatically generalised to any other no. of sources
    
    if not convergence_stable[0]: # No plot if largest estimate hasn't converged (tends to converge first)
        return fig_l_er, ax_l_er

    x_1 = belief_state_mean_over_time[:, 1]
    y_1 = belief_state_mean_over_time[:, 2]

    x_2 = belief_state_mean_over_time[:, 4]
    y_2 = belief_state_mean_over_time[:, 5]   

    x_3 = belief_state_mean_over_time[:, 7]
    y_3 = belief_state_mean_over_time[:, 8]

    dist1 = torch.sqrt((x_1 - source1.x()) ** 2 + (y_1 - source1.y()) ** 2) # Calculate distance between estimate and true source
    dist2 = torch.sqrt((x_2 - source2.x()) ** 2 + (y_2 - source2.y()) ** 2)
    dist3 = torch.sqrt((x_3 - source3.x()) ** 2 + (y_3 - source3.y()) ** 2)

    ax_l_er.cla()  # Clear the current axis

    if show_result:
        ax_l_er.set_title('Result')
    else:
        ax_l_er.set_title('Training...')

    ax_l_er.set_xlabel('Episode')
    ax_l_er.set_ylabel('Distance (m)')

    # Plot lines (only when each source estimate has converged)
    ax_l_er.plot(plot_x_1, dist1[episode_converged[0]:].numpy(), label="Source 1", color="cyan") # Explicitly cast to int for slicing

    if convergence_stable[1]:
       ax_l_er.plot(plot_x_2, dist2[episode_converged[1]:].numpy(), label="Source 2", color="magenta")

    if convergence_stable[2]:
        ax_l_er.plot(plot_x_3, dist3[episode_converged[2]:].numpy(), label="Source 3", color="yellow")
    
    ax_l_er.legend()

    # Pause for dynamic plotting
    plt.pause(0.001)

    # Return the figure and axis objects for external access
    return fig_l_er, ax_l_er

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
            belief_state_mean_np = belief_state_mean.cpu().numpy()  # Ensure it's on the CPU before converting
            goal_chosen = random.randint(0, ((np.sum(~np.isnan(belief_state_mean_np))) - 1)/3 - 1) # Finds out the no. of sources of belief state then chooses random source
            goal = [belief_state_mean[3*goal_chosen + 1], belief_state_mean[3*goal_chosen + 2]]


    for t in count():

        # Calculate total radiation level (from all sources)

        total_radiation_level = source1.radiation_level(agent.x(), agent.y()) + source2.radiation_level(agent.x(), agent.y()) + source3.radiation_level(agent.x(), agent.y())
        total_radiation_level += np.random.normal(loc=0, scale=sd_noise_pct*total_radiation_level) # Add sensor noise to total reading

        # Update/ resample particles in PF
        likelihood = pf.likelihood(agent.x(), agent.y(), particles, total_radiation_level, sd_noise_pct, min_radiation_level, max_radiation_level) # Compute likelihood of each particle

        weights = pf.update_weights(weights, likelihood) # Update weights according to likelihood

        particles, weights, need_resample = pf.resampling_simple(particles, weights, min_no_sources, max_no_sources, min_radiation_level, max_radiation_level, EPS_START, EPS_END, EPS_DECAY, steps_done) # Resample if needed

        if need_resample:
            particles = pf.sort_sources_by_strength(particles) # Only need to re-sort if resampling occurs (weight changes do not affect particle parameters)

        action = select_action(state, goal)

        i = int(action.item())
        actions[i]()

        observation = agent.state()
        reward = -0.02

# Terminate on both the convergence of source estimation/ close to source term estimate

        if np.linalg.norm(np.array([belief_state_mean[1], belief_state_mean[2]]) - np.array([agent.x(), agent.y()])) <= 2.0 and convergence_stable[0] == True: # Terminate episode if agent thinks it is within 2 meters of source, and only if the STE converged already
            terminated = True
            truncated = False
            reward = 0.175*total_radiation_level # Big reward for reaching source estimate (depends on actual radiation level recorded so false positives are not rewarded)
            print(f"Reward: {reward:.3f} (reached within 2 m of source estimate in {agent.count()} steps)")
        elif agent.count() >= 150: # changed from 300 for goal-based
            terminated = False
            truncated = True
            # Initially estimate might be inaccurate but is still far away from starting location, so it motivates agent to leave starting location
            reward = 0.05*(53.15 - np.linalg.norm(np.array([agent.x(), agent.y()]) - np.array([belief_state_mean[1], belief_state_mean[2]]))) # Based on distance to largest state estimate (53.15 is maximum distance from source)
            print(f"Reward: {reward:.3f} (episode reached 150 steps)")
        elif agent.actionPossible() == False: # Discourage agent from attempting to exit search area
            terminated = False
            truncated = False
            reward = -0.0025*belief_state_mean[3] # Was -0.025 (reduced to prevent agent from only aiming to avoid the edge), and not to terminate episode so that somehow crashing into the edge is a better idea than not crashing and moving towards goal
        else:
            terminated = False
            truncated = False

        reward = torch.tensor([reward], device=device)

        done = terminated or truncated # Check if episode is complete

        if ((i_episode + 1) % displaySimulation_p == 0 or (i_episode + 1) == 1) and displaySimulation: # Only show simulation of episodes which are multiples of 100, and the first episode

            for artist in ax_sim.get_children(): # Only clear data points so that radiation map only needs to be plot once
                if isinstance(artist, plt.Line2D) or isinstance(artist, collections.PathCollection):
                    artist.remove()

            # Plot the radiation map once
            if not map_plotted:
                contour = ax_sim.contourf(rad_X, rad_Y, Z_clipped, levels=np.logspace(np.log10(0.01), np.log10(1000), num=400), cmap='viridis', norm=LogNorm(vmin=0.01, vmax=1000), zorder=0)
                
                cbar = fig_sim.colorbar(contour, ax=ax_sim) # Add colorbar with log scale
                cbar.set_label('Equivalent dose rate (mSv/h)')
                cbar.set_ticks([0.01, 0.1, 1, 10, 100, 1000])  # Log ticks

                map_plotted = True # only plot for first time

            src_1 = ax_sim.scatter(source1.x(), source1.y(), marker='*', s=200, edgecolors='red', facecolors='none', zorder=3, label=r'$I_0 = 200$') # Strongest source
            src_2 = ax_sim.scatter(source2.x(), source2.y(), marker='*', s=200, edgecolors='black', facecolors='none', zorder=3, label=r'$I_0 = 140$') # Source 2
            src_3 = ax_sim.scatter(source3.x(), source3.y(), marker='*', s=200, edgecolors='white', facecolors='none', zorder=3, label=r'$I_0 = 120$') # Source 3

            if not legend_plotted:
                legend1 = ax_sim.legend(loc='lower left', bbox_to_anchor=(0, 0.2), title='True sources', fontsize=8, title_fontsize=8) # Add first legend for actual source locations
                ax_sim.add_artist(legend1)

                legend_plotted = True

            # Append the agent's current position to the path
            agent_path.append((agent.x(), agent.y()))

            # Show the path of the agent
            for pos in agent_path:
                path = ax_sim.plot(pos[0], pos[1], marker='o', color='darkgreen', markersize=2, zorder=2)

            # Show the agent's current position as a distinct marker
            pos = ax_sim.plot(agent.x(), agent.y(), marker='x', color='darkgreen', markersize=6, zorder=2)

            colors = ['cyan', 'orange', 'lime', 'yellow', 'pink'] # To distinguish between source prediction number in plotting

            # Show particles in particle filter (color coded by source prediction no.)

            src_handles = []
            src_labels = []
            
            for i in range(max_no_sources):
                # Select columns corresponding to specific source prediction
                source_prediction = particles[:, [3*i + 1, 3*i + 2]]

                # Filter rows where neither column has NaN
                filtered_data = source_prediction[~np.isnan(source_prediction).any(axis=1)]

                if not len(filtered_data) == 0:
                    src = ax_sim.scatter(filtered_data[:, 0], filtered_data[:, 1], marker='.', s=2, color=colors[i], label=f'{i+1}', zorder=2)
                    src_handles.append(src)
                    src_labels.append(f'{i+1}')
            
            # Pause briefly to update the plot

            ax_sim.legend(handles = src_handles, labels = src_labels, title="Estimated sources", loc='lower left', bbox_to_anchor=(0, 0), markerscale=4, fontsize=8, title_fontsize=8) # Show estimated legend
            
            ax_sim.set_xlim(0, search_area_x)
            ax_sim.set_ylim(0, search_area_y)
            plt.pause(0.000001)

            if savePlot and done: # Save .png image of simulation plot

                sub_dir = "multi_source_results"
                sim_sub_dir = "simulation_results"

                full_dir = os.path.join(sub_dir, sim_sub_dir)

                # Ensure the subdirectory exists
                os.makedirs(sub_dir, exist_ok=True)

                os.makedirs(full_dir, exist_ok=True)

                fig_sim.savefig(os.path.join(full_dir, f"episode_{i_episode + 1}"), bbox_inches='tight')

        if done:
            next_state = None

            all_STE_x_mean, all_STE_y_mean, all_STE_strength_mean, all_STE_x_var, all_STE_y_var, all_STE_strength_var,= pf.estimate(particles, weights) # Fetch belief state for episode

            num_sources_int = particles[:,0].astype(int)

            bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5] # 6 bin edges for source no. prediction counts from 1 to 5

            source_counts, bins = np.histogram(num_sources_int, bins=bins)


            # ADD BELIEF STATE INTO MOVING AVERAGE
            non_avg_belief_state_mean = []

            # First value: the index of the max value of source_counts, +1 (since it's 1-indexed)
            non_avg_belief_state_mean.append(np.argmax(source_counts) + 1)

            # For each source, add x, y, and strength values in the order x, y, strength
            for i in range(max_no_sources):
                non_avg_belief_state_mean.append(all_STE_x_mean[i])
                non_avg_belief_state_mean.append(all_STE_y_mean[i])
                non_avg_belief_state_mean.append(all_STE_strength_mean[i])

            # Convert to a tensor
            non_avg_belief_state_mean = torch.tensor(non_avg_belief_state_mean, dtype=torch.float32)

            # Compute moving average with 'window_size' no. of values or all available values.
            moving_average_queue_mean.append(non_avg_belief_state_mean)

            belief_state_mean = torch.mean(torch.stack(list(moving_average_queue_mean)), dim=0)


            # ADD VARIANCE OF BELIEF STATE INTO MOVING AVERAGE
            non_avg_belief_state_var = []

            # First value: variance of source_counts (variance in belief of source no.)
            source_no_beliefs = np.arange(min_no_sources, max_no_sources + 1) # Initialise array with source belief no.
            non_avg_belief_state_var.append(DescrStatsW(source_no_beliefs, weights=source_counts).std)

            # For each source, add x, y, and strength values in the order x, y, strength
            for i in range(max_no_sources):
                non_avg_belief_state_var.append(all_STE_x_var[i])
                non_avg_belief_state_var.append(all_STE_y_var[i])
                non_avg_belief_state_var.append(all_STE_strength_var[i])

            # Convert to a tensor
            non_avg_belief_state_var = torch.tensor(non_avg_belief_state_var, dtype=torch.float32)

            # Compute moving average with 'window_size' no. of values or all available values.
            moving_average_queue_var.append(non_avg_belief_state_var)

            belief_state_var = torch.mean(torch.stack(list(moving_average_queue_var)), dim=0)


            # Determine convergence with variance of particle estimates

            # Convergence of estimates is defined by set tolerances on standard deviation
            tolerance_loc = 6.0
            tolerance_str = 6.0
            tolerance_no = 0.60

            
            # CHECK CONVERGENCE STABILITY OF SOURCE 1
            if (np.sqrt(belief_state_var[1].item()) < tolerance_loc and    # Ensure all tolerances are met 
                np.sqrt(belief_state_var[2].item()) < tolerance_loc and 
                np.sqrt(belief_state_var[3].item()) < tolerance_str and 
                np.sqrt(belief_state_var[0].item()) < tolerance_no):

                if not convergence_stable[0]:
                    convergence_tracking[0] += 1
                    
                    if convergence_tracking[0] >= 25: # Only recognise stability of convergence when converged 25 episodes in a row
                        episode_converged[0] = i_episode
                        convergence_stable[0] = True

            else:
                convergence_tracking[0] = 0 # Reset convergence streak if broken

            if convergence_stable[0]:
                plot_x_1.append(i_episode) # Keep an array of all episodes since convergence for plotting

            
            # CHECK CONVERGENCE STABILITY OF SOURCE 2
            if (np.sqrt(belief_state_var[4].item()) < tolerance_loc and    # Ensure all tolerances are met 
                np.sqrt(belief_state_var[5].item()) < tolerance_loc and 
                np.sqrt(belief_state_var[6].item()) < tolerance_str):

                if not convergence_stable[1]:
                    convergence_tracking[1] += 1
                    
                    if convergence_tracking[1] >= 25: # Only recognise stability of convergence when converged 25 episodes in a row
                        episode_converged[1] = i_episode
                        convergence_stable[1] = True

            else:
                convergence_tracking[1] = 0 # Reset convergence streak if broken

            if convergence_stable[1]:
                plot_x_2.append(i_episode) # Keep an array of all episodes since convergence for plotting

            
            # CHECK CONVERGENCE STABILITY OF SOURCE 3
            if (np.sqrt(belief_state_var[7].item()) < tolerance_loc and    # Ensure all tolerances are met 
                np.sqrt(belief_state_var[8].item()) < tolerance_loc and 
                np.sqrt(belief_state_var[9].item()) < tolerance_str):

                if not convergence_stable[2]:
                    convergence_tracking[2] += 1
                    
                    if convergence_tracking[2] >= 25: # Only recognise stability of convergence when converged 25 episodes in a row
                        episode_converged[2] = i_episode
                        convergence_stable[2] = True

            else:
                convergence_tracking[2] = 0 # Reset convergence streak if broken

            if convergence_stable[2]:
                plot_x_3.append(i_episode) # Keep an array of all episodes since convergence for plotting

            print(f"Estimated number of sources: {belief_state_mean[0]:.2f} ± {np.sqrt(belief_state_var[0].item()):.2f}, Goal chosen for episode: ({goal[0]:.2f}, {goal[1]:.2f})")
            print(f"Estimated strongest x: {belief_state_mean[1]:.2f} ± {np.sqrt(belief_state_var[1].item()):.2f}, Estimated strongest y: {belief_state_mean[2]:.2f} ± {np.sqrt(belief_state_var[2].item()):.2f}, Estimated strongest strength: {belief_state_mean[3]:.2f} ± {np.sqrt(belief_state_var[3].item()):.2f}") # Equivalent to range of 2 std devs for ~95% confidence
            print(f"Convergence stability: {'True' if convergence_stable[0] else 'False'} (converged {int(convergence_tracking[0])} episodes in a row)\n")

            for i in range(np.argmax(source_counts)):
                print(f"Estimated source {i+2} x: {belief_state_mean[3*(i + 1) + 1]:.2f} ± {np.sqrt(belief_state_var[3*(i + 1) + 1].item()):.2f}, Estimated source {i+2} y: {belief_state_mean[3*(i + 1) + 2]:.2f} ± {np.sqrt(belief_state_var[3*(i + 1) + 2].item()):.2f}, Estimated source {i+2} strength: {belief_state_mean[3*(i + 1) + 3]:.2f} ± {np.sqrt(belief_state_var[3*(i + 1) + 3].item()):.2f}")

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

            belief_state_mean = belief_state_mean.unsqueeze(0) # To turn current_estimate into a 2D tensor
            belief_state_mean_over_time = torch.cat((belief_state_mean_over_time, belief_state_mean), dim=0)
            belief_state_mean = belief_state_mean.flatten(0) # To turn current_estimate back into a 1D tensor

            if displayPlot:
                fig_d, ax_d = plot_distance()
                fig_l, ax_l = plot_length()
                fig_l_e, ax_l_e = plot_loc_estimate()
                fig_s_e, ax_s_e = plot_strength_estimate_error()
                fig_n_e, ax_n_e  = plot_number_estimate()
                fig_l_er, ax_l_er = plot_loc_error()
            break


print('Complete')

# Indicate training is complete and show results
plot_distance(show_result=True)
plot_length(show_result=True)
plot_loc_estimate(show_result=True)
plot_strength_estimate_error(show_result=True)
plot_number_estimate(show_result=True)
plot_loc_error(show_result=True)

if savePlot: # Save plot object as tuple in external file using pickle, as well as pngs of plotted graphs

    current_dir = os.getcwd()
    sub_dir = "multi_source_results"

    # Ensure the subdirectory exists
    os.makedirs(sub_dir, exist_ok=True)


    # Pickle objects for plotting
    with open(os.path.join(sub_dir, "distance_plot.pkl"), "wb") as file:
        pickle.dump((fig_d, ax_d), file)

    with open(os.path.join(sub_dir,"length_plot.pkl"), "wb") as file:
        pickle.dump((fig_l, ax_l), file)

    with open(os.path.join(sub_dir,"location_plot.pkl"), "wb") as file:
        pickle.dump((fig_l_e, ax_l_e), file)

    with open(os.path.join(sub_dir,"strength_error_plot.pkl"), "wb") as file:
        pickle.dump((fig_s_e, ax_s_e), file)

    with open(os.path.join(sub_dir,"number_plot.pkl"), "wb") as file:
        pickle.dump((fig_n_e, ax_n_e), file)

    with open(os.path.join(sub_dir,"error_plot.pkl"), "wb") as file:
        pickle.dump((fig_l_er, ax_l_er), file)

    # .pngs of finished plots
    try:
        fig_d.savefig(os.path.join(sub_dir,"dist"), bbox_inches='tight')
        fig_l.savefig(os.path.join(sub_dir,"length"), bbox_inches='tight')
        fig_l_e.savefig(os.path.join(sub_dir,"coords"), bbox_inches='tight')
        fig_s_e.savefig(os.path.join(sub_dir,"est_strengths"), bbox_inches='tight')
        fig_n_e.savefig(os.path.join(sub_dir,"num_sources"), bbox_inches='tight')
        fig_l_er.savefig(os.path.join(sub_dir,"estimation_error"), bbox_inches='tight')
    except:
        print("Couldn't save .png figures!")

    folder_path = os.path.join(current_dir, sub_dir)
    print(f"Files saved at: {folder_path}")

plt.ioff()
plt.show()
