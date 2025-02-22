# -*- coding: utf-8 -*-
"""ddpg.py

This script implements the Deep Deterministic Policy Gradient (DDPG) algorithm for continuous action spaces, as part of a Reinforcement Learning course.
"""

import gym
import numpy as np
from collections import deque
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Ornstein-Uhlenbeck Noise
# ----------------------------------------------------------------------
class OUNoise(object):
    """
    Ornstein-Uhlenbeck process for generating noise to explore action space.
    """
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        """
        Initializes the OUNoise process.

        Args:
            action_space (gym.spaces.Box): The action space of the environment.
            mu (float): The mean of the process.
            theta (float): The rate of mean reversion.
            max_sigma (float): The maximum standard deviation of the process.
            min_sigma (float): The minimum standard deviation of the process.
            decay_period (int): The number of steps over which the standard deviation decays.
        """
        self.mu = mu # Mean of the process
        self.theta = theta # Rate of mean reversion
        self.sigma = max_sigma # Current standard deviation
        self.max_sigma = max_sigma # Maximum standard deviation
        self.min_sigma = min_sigma # Minimum standard deviation
        self.decay_period = decay_period # Decay period
        self.action_dim = action_space.shape[0] # Dimension of the action space
        self.low = action_space.low # Lower bound of the action space
        self.high = action_space.high # Upper bound of the action space
        self.reset() # Reset the process

    def reset(self):
        """
        Resets the state of the process to the mean.
        """
        self.state = np.ones(self.action_dim) * self.mu # Reset state to the mean

    def evolve_state(self):
        """
        Evolves the state of the process according to the OU equation.
        """
        x = self.state # Current state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim) # OU equation
        self.state = x + dx # Update state
        return self.state # Return the new state

    def get_action(self, action, t=0):
        """
        Adds noise to the given action.

        Args:
            action (np.ndarray): The action to add noise to.
            t (int): The current timestep.

        Returns:
            np.ndarray: The noisy action, clipped to the action space bounds.
        """
        ou_state = self.evolve_state() # Get the OU state
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period) # Decay the standard deviation

        return np.clip(action + ou_state, self.low, self.high) # Clip the noisy action to the action space bounds

# ----------------------------------------------------------------------
# Action Normalization Wrapper
# ----------------------------------------------------------------------
class NormalizedEnv(gym.ActionWrapper):
    """
    Action wrapper that normalizes actions to the range [-1, 1].
    """
    def _action(self, action):
        """
        Transforms the action to the environment's action space.
        """
        act_k = (self.action_space.high - self.action_space.low) / 2. # Scaling factor
        act_b = (self.action_space.high + self.action_space.low) / 2.  # Bias
        return act_k * action + act_b # Apply the transformation

    def _reverse_action(self, action):
        """
        Reverses the transformation to get the original action.
        """
        act_k_inv = 2.0 / (self.action_space.high - self.action_space.low) # Inverse scaling factor
        act_b = (self.action_space.high + self.action_space.low) / 2.  # Bias
        return act_k_inv * (action - act_b) # Apply the inverse transformation

# ----------------------------------------------------------------------
# Replay Memory
# ----------------------------------------------------------------------
class Memory:
    """
    Replay memory buffer for storing experiences.
    """
    def __init__(self, max_size):
        """
        Initializes the Memory buffer.

        Args:
            max_size (int): The maximum number of experiences to store.
        """
        self.max_size = max_size  # Maximum buffer size
        self.buffer = deque(maxlen=max_size) # Use deque for efficient appending and popping

    def push(self, state, action, reward, next_state, done):
        """
        Adds an experience to the buffer.

        Args:
            state (np.ndarray): The current state.
            action (np.ndarray): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state.
            done (bool): Whether the episode is done.
        """
        experience = (state, action, np.array([reward]), next_state, done) # Create the experience tuple
        self.buffer.append(experience) # Add the experience to the buffer

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            tuple: A tuple of lists containing the sampled states, actions, rewards, next states, and done flags.
        """
        state_batch = [] # List to store the states
        action_batch = [] # List to store the actions
        reward_batch = [] # List to store the rewards
        next_state_batch = [] # List to store the next states
        done_batch = []  # List to store the done flags

        batch = random.sample(self.buffer, batch_size) # Randomly sample a batch of experiences

        # Unpack the batch into separate lists
        for experience in batch:
            state, action, reward, next_state, done = experience # Unpack the experience
            state_batch.append(state) # Append the state
            action_batch.append(action) # Append the action
            reward_batch.append(reward) # Append the reward
            next_state_batch.append(next_state) # Append the next state
            done_batch.append(done) # Append the done flag

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch # Return the lists

    def __len__(self):
        """
        Returns the current size of the buffer.
        """
        return len(self.buffer) # Return the number of experiences in the buffer

# ----------------------------------------------------------------------
# Critic Network
# ----------------------------------------------------------------------
class Critic(nn.Module):
    """
    Critic network for estimating the Q-value function.
    """
    def __init__(self, input_size, hidden_size, output_size=1):
        """
        Initializes the Critic network.

        Args:
            input_size (int): The number of input features (state dimension + action dimension).
            hidden_size (int): The size of the hidden layers.
            output_size (int): The number of output values (Q-value).
        """
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # First linear layer
        self.linear2 = nn.Linear(hidden_size, hidden_size) # Second linear layer
        self.linear3 = nn.Linear(hidden_size, output_size) # Output layer

    def forward(self, state, action):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): The input state.
            action (torch.Tensor): The input action.

        Returns:
            torch.Tensor: The estimated Q-value.
        """
        x = torch.cat([state, action], 1) # Concatenate state and action
        x = F.relu(self.linear1(x)) # First layer with ReLU activation
        x = F.relu(self.linear2(x)) # Second layer with ReLU activation
        x = self.linear3(x) # Output layer (Q-value)

        return x # Return the Q-value

# ----------------------------------------------------------------------
# Actor Network
# ----------------------------------------------------------------------
class Actor(nn.Module):
    """
    Actor network for generating actions.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the Actor network.

        Args:
            input_size (int): The number of input features (state dimension).
            hidden_size (int): The size of the hidden layers.
            output_size (int): The number of output values (action dimension).
        """
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # First linear layer
        self.linear2 = nn.Linear(hidden_size, hidden_size) # Second linear layer
        self.linear3 = nn.Linear(hidden_size, output_size) # Output layer

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            torch.Tensor: The generated action.
        """
        x = F.relu(self.linear1(state)) # First layer with ReLU activation
        x = F.relu(self.linear2(x)) # Second layer with ReLU activation
        x = torch.tanh(self.linear3(x)) # Output layer (action) with tanh activation

        return x # Return the action

# ----------------------------------------------------------------------
# DDPG Agent
# ----------------------------------------------------------------------
class DDPGagent:
    """
    DDPG agent for continuous control tasks.
    """
    def __init__(self, env, hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=50000):
        """
        Initializes the DDPG agent.

        Args:
            env (gym.Env): The environment to train in.
            hidden_size (int): The size of the hidden layers.
            actor_learning_rate (float): The learning rate for the actor network.
            critic_learning_rate (float): The learning rate for the critic network.
            gamma (float): The discount factor.
            tau (float): The soft update coefficient.
            max_memory_size (int): The maximum size of the replay memory.
        """
        # Params
        self.num_states = env.observation_space.shape[0] # Number of states
        self.num_actions = env.action_space.shape[0] # Number of actions
        self.gamma = gamma # Discount factor
        self.tau = tau  # Soft update coefficient

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions) # Create the actor network
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions) # Create the target actor network
        self.critic = Critic(self.num_states + self.num_actions, hidden_size) # Create the critic network
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size) # Create the target critic network

        # Initialize target networks with the same weights as the main networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data) # Copy the weights

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data) # Copy the weights

        # Training
        self.memory = Memory(max_memory_size)  # Create the replay memory
        self.critic_criterion = nn.MSELoss()  # Create the critic loss function (MSE loss)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate) # Create the actor optimizer (Adam)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate) # Create the critic optimizer (Adam)

    def get_action(self, state):
        """
        Selects an action based on the current state.

        Args:
            state (np.ndarray): The current state.

        Returns:
            np.ndarray: The action to take.
        """
        state = Variable(torch.from_numpy(state).float().unsqueeze(0)) # Convert state to a PyTorch Variable
        action = self.actor.forward(state)  # Get the action from the actor network
        action = action.detach().numpy()[0]  # Convert the action to a NumPy array

        return action  # Return the action

    def update(self, batch_size):
        """
        Updates the actor and critic networks based on a sampled batch of experiences.

        Args:
            batch_size (int): The number of experiences to sample.
        """
        # Sample a batch of experiences from the replay memory
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)

        # Convert the sampled data to PyTorch tensors
        states = torch.FloatTensor(states) # States
        actions = torch.FloatTensor(actions) # Actions
        rewards = torch.FloatTensor(rewards) # Rewards
        next_states = torch.FloatTensor(next_states) # Next states

        # --------------------------------------------------------------
        # Critic Update
        # --------------------------------------------------------------
        # Calculate the Q-values for the current states and actions
        Qvals = self.critic.forward(states, actions) # Q(s, a)

        # Calculate the actions for the next states using the target actor network
        next_actions = self.actor_target.forward(next_states) # mu'(s')

        # Calculate the Q-values for the next states and actions using the target critic network
        next_Q = self.critic_target.forward(next_states, next_actions.detach()) # Q'(s', mu'(s'))

        # Calculate the expected Q-values using the Bellman equation
        Qprime = rewards + self.gamma * next_Q # r + gamma * Q'(s', mu'(s'))

        # Calculate the critic loss using mean squared error
        critic_loss = self.critic_criterion(Qvals, Qprime) # MSE(Q(s, a), r + gamma * Q'(s', mu'(s')))

        # Update the critic network
        self.critic_optimizer.zero_grad() # Clear the gradients
        critic_loss.backward() # Calculate the gradients
        self.critic_optimizer.step() # Update the weights

        # --------------------------------------------------------------
        # Actor Update
        # --------------------------------------------------------------
        # Calculate the policy loss by taking the negative mean of the Q-values obtained by passing the states through the actor and critic networks
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean() # -E[Q(s, mu(s))]

        # Update the actor network
        self.actor_optimizer.zero_grad() # Clear the gradients
        policy_loss.backward() # Calculate the gradients
        self.actor_optimizer.step() # Update the weights

        # --------------------------------------------------------------
        # Target Network Updates (Soft Updates)
        # --------------------------------------------------------------
        # Soft update the target networks by blending the main network weights with the target network weights
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau)) # theta' = tau * theta + (1 - tau) * theta'

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau)) # theta' = tau * theta + (1 - tau) * theta'

# ----------------------------------------------------------------------
# Main Script
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Create the environment
    #env = gym.make("Pendulum-v0") # Pendulum environment
    #env = gym.make("LunarLanderContinuous-v2") # LunarLander environment
    env = gym.make("MountainCarContinuous-v0")  # MountainCar environment

    # Create the DDPG agent
    agent = DDPGagent(env)

    # Create the OU noise process
    noise = OUNoise(env.action_space)

    # Training parameters
    batch_size = 128 # Batch size for training
    num_episodes = 100 # Number of episodes to train for

    # Lists to store rewards
    rewards = []  # List to store rewards for each episode
    avg_rewards = [] # List to store average rewards over the last 10 episodes

    # Training loop
    for episode in range(num_episodes):
        # Reset the environment and the noise process
        state = env.reset() # Reset the environment
        noise.reset() # Reset the noise process
        episode_reward = 0 # Initialize the episode reward

        # Run the episode
        for step in range(500):
            # Get the action from the agent and add noise for exploration
            action = agent.get_action(state) # Get the action from the agent
            action = noise.get_action(action, step)  # Add noise to the action

            # Take a step in the environment
            new_state, reward, done, _ = env.step(action)

            # Store the experience in the replay memory
            agent.memory.push(state, action, reward, new_state, done) # Store the experience

            # Update the agent if there are enough experiences in the replay memory
            if len(agent.memory) > batch_size:
                agent.update(batch_size) # Update the agent

            # Update the state and the episode reward
            state = new_state # Update the state
            episode_reward += reward  # Update the episode reward

            # If the episode is done, print the results and break the loop
            if done:
                sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:]))) # Print the episode results
                break

        # Store the episode reward and calculate the average reward over the last 10 episodes
        rewards.append(episode_reward) # Store the episode reward
        avg_rewards.append(np.mean(rewards[-10:])) # Calculate the average reward

    # Plot the results
    plt.plot(rewards, label="Rewards") # Plot the rewards
    plt.plot(avg_rewards, label="Average Rewards") # Plot the average rewards
    plt.xlabel('Episode') # Set the x-axis label
    plt.ylabel('Reward') # Set the y-axis label
    plt.title("DDPG Training") # Set the title
    plt.legend() # Add the legend
    plt.show() # Show the plot
