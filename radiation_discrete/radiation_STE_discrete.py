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

# Turns on interactive mode for MATLAB plots, so that plot is showed without use of plt.show()
plt.ion()

# Initialising environment

# Initialise search area
search_area_x = 50
search_area_y = 50

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

particles = pf.create_uniform_particles((0,search_area_x), (0,search_area_y), (radiation_level-50, radiation_level+150), N) # Initialise particles randomly in search area

weights = np.ones(N) / N # All weights initalised equally

print(weights)

device = torch.device( # Checks whether CUDA/MPS device is available for acceleration, otherwise cpu is used.
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print(f"Using {device} device")

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

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.00
EPS_DECAY = 50
TAU = 0.005
LR = 0.001

n_actions = len(actions)
n_observations = (search_area_x + 1) * (search_area_y + 1)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)  # Changed from (1, 16) to (1, 1)
    else:
        return torch.tensor([[random.randint(0,3)]], device=device, dtype=torch.long) # 1-4 represents the index of action in action array

episode_end_distances = []

def plot_distance(show_result=False):
    plt.figure(1)
    distances_t = torch.tensor(episode_end_distances, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Distance to source at the end of episode')
    plt.plot(distances_t.numpy())

    if len(distances_t) >= 100:
        means = distances_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)

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

num_episodes = 150 if torch.cuda.is_available() or torch.backends.mps.is_available() else 50

def one_hot_encode(state, n_observations):
    one_hot = torch.zeros(n_observations, dtype=torch.float32, device=device)
    one_hot[state] = 1
    return one_hot

for i_episode in range(num_episodes):
    agent.reset()
    state = agent.state()
    state = one_hot_encode(state, n_observations).unsqueeze(0)

    particles = pf.create_uniform_particles((0,search_area_x), (0,search_area_y), (radiation_level-50, radiation_level+150), N) # Initialise particles randomly in search area

    weights = np.ones(N) / N # All weights initalised equally

    for t in count():
        action = select_action(state)
        i = int(action.item())
        actions[i]()

        observation = agent.state()
        reward = -0.1
        # reward = 0.01 * source.radiation_level(agent.x(), agent.y())

        # Update/ resample particles in PF
        likelihood = pf.likelihood(agent.x(), agent.y(), particles, source.radiation_level(agent.x(), agent.y()), sd_noise) # Compute likelihood of each particle

        weights = pf.update_weights(weights, likelihood) # Update weights according to likelihood

        particles, weights = pf.resampling(particles, weights) # Resample if needed

        est_mean, est_var = pf.estimate(particles, weights) # Fetch estimate of source location/ strength

        if source.radiation_level(agent.x(), agent.y()) >= 4: # Terminate episode if within 5 meters of source
            terminated = True
            reward = 10
            print(f"Reward: {reward:.2f}")
        else:
            terminated = False

        if agent.count() >= 100:
            truncated = True
            reward = 15 - math.sqrt(radiation_level / source.radiation_level(agent.x(), agent.y()))
            print(f"Reward: {reward:.2f}")
        else:
            truncated = False
            # reward -= 0.001 * agent.count()

        reward = torch.tensor([reward], device=device)

        if i_episode % 50 == 0: # Only show simulation of episodes which are multiples of 50 (including 0).
            plt.figure(2)
            plt.clf()  # Clear the figure
    
            # Re-plot the radiation map and radiation source
            plt.contourf(rad_X, rad_Y, Z_clipped, 200, cmap='viridis')
            plt.colorbar()
            plt.plot(source.x(), source.y(), 'ro', markersize=4)
    
            # Show agent as green point (current position only)
            plt.plot(agent.x(), agent.y(), marker='x', color=(0.2, 0.8, 0), markersize=4)

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
            print(f"Estimated source x: {est_mean[0]:.2f}, Estimated source y: {est_mean[1]:.2f}, Estimated source strength: {est_mean[2]:.2f}")

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
            plot_distance()
            break


print('Complete')
plot_distance(show_result=True)
plt.ioff()
plt.show()
