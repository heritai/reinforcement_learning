# -*- coding: utf-8 -*-
"""dqn.py

This script implements the DQN algorithm for solving CartPole, GridWorld, and LunarLander environments, as part of a Reinforcement Learning course (TME4).
"""

import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a Transition namedtuple for storing experience replay data
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# ----------------------------------------------------------------------
# Replay Memory
# ----------------------------------------------------------------------
class ReplayMemory(object):
    """
    Replay Memory class for storing and sampling experiences.
    """
    def __init__(self, capacity):
        """
        Initializes the Replay Memory.

        Args:
            capacity (int): The maximum number of transitions to store.
        """
        self.capacity = capacity
        self.memory = [] # List to store the transitions
        self.position = 0 # Pointer to the next location to write to

    def push(self, *args):
        """
        Saves a transition to memory.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None) # Allocate space if needed
        self.memory[self.position] = Transition(*args) # Store the transition
        self.position = (self.position + 1) % self.capacity # Circular buffer

    def sample(self, batch_size):
        """
        Samples a random batch of transitions from memory.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            list: A list of randomly sampled transitions.
        """
        return random.sample(self.memory, batch_size) # Randomly sample transitions

    def __len__(self):
        """
        Returns the number of transitions currently stored in memory.
        """
        return len(self.memory)

# ----------------------------------------------------------------------
# DQN Network
# ----------------------------------------------------------------------
class DQN(nn.Module):
    """
    Deep Q-Network (DQN) class.
    """
    def __init__(self, inSize, outSize, layers=[]):
        """
        Initializes the DQN.

        Args:
            inSize (int): The input size (state dimension).
            outSize (int): The output size (number of actions).
            layers (list): A list of hidden layer sizes.
        """
        super(DQN, self).__init__()
        self.layers = nn.ModuleList([]) # Use ModuleList for storing layers

        # Create the hidden layers
        for x in layers:
            self.layers.append(nn.Linear(inSize, x)) # Add a linear layer
            inSize = x  # Update the input size for the next layer

        # Create the output layer
        self.layers.append(nn.Linear(inSize, outSize)) # Add the output layer

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor (state).

        Returns:
            torch.Tensor: The output tensor (Q-values).
        """
        x = self.layers[0](x)  # Input layer
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.leaky_relu(x)  # Apply leaky ReLU activation
            x = self.layers[i](x) # Hidden layers
        return x # Output Q-values

# ----------------------------------------------------------------------
# Feature Extractor (for GridWorld)
# ----------------------------------------------------------------------
class FeaturesExtractor(object):
    """
    Feature Extractor class for GridWorld environment. Extracts features from the raw observation.
    """
    def __init__(self, outSize):
        """
        Initializes the FeaturesExtractor.

        Args:
            outSize (int): The size of each feature map.
        """
        super().__init__()
        self.outSize = outSize * 3 # Three feature maps (agent, yellow, rose)

    def getFeatures(self, obs):
        """
        Extracts features from the given observation.

        Args:
            obs (np.ndarray): The raw observation from the GridWorld environment.

        Returns:
            np.ndarray: The extracted features.
        """
        state = np.zeros((3, np.shape(obs)[0], np.shape(obs)[1])) # Create empty feature maps
        state[0] = np.where(obs == 2, 1, state[0]) # Agent position
        state[1] = np.where(obs == 4, 1, state[1]) # Yellow elements
        state[2] = np.where(obs == 6, 1, state[2]) # Rose elements
        return state.reshape(1, -1) # Flatten the feature maps

# ----------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------
BATCH_SIZE = 128  # Batch size for training
GAMMA = 0.999  # Discount factor
EPS_START = 0.9  # Exploration rate start value
EPS_END = 0.05  # Exploration rate end value
EPS_DECAY = 200 # Exploration rate decay rate
TARGET_UPDATE = 20  # Update target network frequency

# ----------------------------------------------------------------------
# Training Loop
# ----------------------------------------------------------------------
def train(env, n_episodes=1000, hidden_layers=[12], feature_extractor=None):
    """
    Trains a DQN agent in the given environment.

    Args:
        env (gym.Env): The environment to train in.
        n_episodes (int): The number of episodes to train for.
        hidden_layers (list): A list of hidden layer sizes for the DQN.
        feature_extractor (FeaturesExtractor): An optional feature extractor for preprocessing the state.

    Returns:
        list: A list of cumulative rewards for each episode.
    """
    # Get environment information
    n_actions = env.action_space.n  # Number of actions

    if feature_extractor is not None:
        # Use the feature extractor to get the state dimension
        state = feature_extractor.getFeatures(env.reset())
        state_dim = state.shape[1]
    else:
        # Use the environment to get the state dimension
        state_dim = len(env.reset())

    # Initialize networks
    policy_net = DQN(state_dim, n_actions, hidden_layers).to(device) # Policy network
    target_net = DQN(state_dim, n_actions, hidden_layers).to(device) # Target network
    target_net.load_state_dict(policy_net.state_dict())  # Initialize target network with policy network weights
    target_net.eval() # Set target network to evaluation mode

    # Initialize optimizer and replay memory
    optimizer = optim.RMSprop(policy_net.parameters()) # RMSprop optimizer
    memory = ReplayMemory(10000) # Replay memory

    steps_done = 0  # Global step counter
    rewards = [] # List to store rewards
    episode_durations = []  # List to store episode durations

    # Define action selection function (epsilon-greedy policy)
    def select_action(state):
        """
        Selects an action using the epsilon-greedy policy.
        """
        nonlocal steps_done # Access the global steps_done variable

        sample = random.random() # Generate a random number
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY) # Calculate the exploration rate
        steps_done += 1 # Increment the step counter

        if sample > eps_threshold:
            # Exploit: Choose the action with the highest Q-value
            with torch.no_grad():
                return policy_net(state).argmax().view(1, 1)  # Returns the index of the maximum Q-value
        else:
            # Explore: Choose a random action
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)  # Returns a random action

    # Define optimization function (DQN update)
    def optimize_model():
        """
        Performs one step of optimization.
        """
        if len(memory) < BATCH_SIZE:
            return # Wait until memory has enough samples

        # Sample a batch from replay memory
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions)) # Transpose the batch

        # Compute a mask of non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool) # Boolean tensor indicating non-terminal states
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])  # Concatenate all non-terminal next states

        # Concatenate batch elements
        state_batch = torch.cat(batch.state) # States
        action_batch = torch.cat(batch.action) # Actions
        reward_batch = torch.cat(batch.reward) # Rewards

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch) # Q-values for the taken actions

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device) # Initialize with zeros
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()  # Get maximum Q-values for next states from target network

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch  # Apply Bellman equation

        # Compute Huber loss (Smooth L1 Loss)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))  # Calculate the loss

        # Optimize the model
        optimizer.zero_grad()  # Clear the gradients
        loss.backward() # Compute the gradients
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # Clip gradients to prevent exploding gradients
        optimizer.step() # Update the weights

    # Main training loop
    for i_episode in range(n_episodes):
        # Initialize the environment and state
        if feature_extractor is not None:
            state = torch.from_numpy(feature_extractor.getFeatures(env.reset())).float().view(1, -1).to(device) # Get initial state and preprocess
        else:
            state = torch.from_numpy(env.reset()).float().view(1, -1).to(device) # Get initial state

        rsum = 0 # Cumulative reward for this episode
        for t in count():
            # Select and perform an action
            action = select_action(state) # Choose an action
            obs, reward, done, _ = env.step(action.item())  # Take a step in the environment
            reward = torch.tensor([reward], device=device) # Convert reward to tensor

            # Observe new state
            if not done:
                if feature_extractor is not None:
                    next_state = torch.from_numpy(feature_extractor.getFeatures(obs)).float().view(1, -1).to(device)  # Preprocess the next state
                else:
                    next_state = torch.from_numpy(obs).float().view(1, -1).to(device) # Convert the next state to tensor
            else:
                next_state = None # Terminal state

            # Store the transition in memory
            memory.push(state, action, next_state, reward)  # Store the transition

            # Move to the next state
            state = next_state  # Update the current state
            rsum += reward  # Update cumulative reward

            # Perform one step of the optimization (on the target network)
            optimize_model()  # Optimize the model

            if done:
                episode_durations.append(t + 1) # Record the episode duration
                rewards.append(rsum.item())  # Append the reward
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict()) # Copy weights from policy net to target net

    print('Complete')
    env.close() # Close environment

    return rewards, episode_durations # Return results

# ----------------------------------------------------------------------
# Plotting Function
# ----------------------------------------------------------------------
def plot_results(rewards):
    """
    Plots the cumulative rewards over episodes.

    Args:
        rewards (list): A list of rewards for each episode.
    """
    plt.plot(np.cumsum(rewards)) # Plot the cumulative rewards
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.title("Cumulative Reward per Episode")
    plt.show()

# ----------------------------------------------------------------------
# Main Script
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # CartPole
    env_cartpole = gym.make('CartPole-v0').unwrapped
    rewards_cartpole, durations_cartpole = train(env_cartpole, n_episodes=1000, hidden_layers=[12])
    plot_results(rewards_cartpole)

    # GridWorld
    import gridworld
    env_gridworld = gym.make("gridworld-v0")
    env_gridworld.setPlan("gridworldPlans/plan4.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1}) # Set up GridWorld
    fex_grid = FeaturesExtractor(1) # Create feature extractor
    rewards_gridworld, durations_gridworld = train(env_gridworld, n_episodes=1000, hidden_layers=[12], feature_extractor=fex_grid) # Pass feature extractor
    plot_results(rewards_gridworld)

    # LunarLander
    env_lunar = gym.make('LunarLander-v2')
    rewards_lunar, durations_lunar = train(env_lunar, n_episodes=1000, hidden_layers=[64, 32]) # Different hidden layer configuration
    plot_results(rewards_lunar)
