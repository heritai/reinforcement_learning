# -*- coding: utf-8 -*-
"""policy_gradient.py

This script implements the Advantage Actor-Critic (A2C) algorithm for policy gradients, as part of a Reinforcement Learning course.
"""

import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------
HIDDEN_SIZE = 256  # Size of the hidden layers in the neural network
LEARNING_RATE = 3e-4 # Learning rate for the optimizer
GAMMA = 0.99 # Discount factor for future rewards
NUM_STEPS = 300 # Maximum number of steps per episode
MAX_EPISODES = 3000  # Maximum number of episodes to train for

# ----------------------------------------------------------------------
# Actor-Critic Network
# ----------------------------------------------------------------------
class ActorCritic(nn.Module):
    """
    Actor-Critic network for policy gradient methods.
    """
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        """
        Initializes the ActorCritic network.

        Args:
            num_inputs (int): Number of input features (state dimension).
            num_actions (int): Number of possible actions.
            hidden_size (int): Size of the hidden layers.
            learning_rate (float): Learning rate for the optimizer.
        """
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions # Store the number of actions

        # Critic network (Value function approximator)
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size) # First linear layer
        self.critic_linear2 = nn.Linear(hidden_size, 1) # Output layer (single value)

        # Actor network (Policy function approximator)
        self.actor_linear1 = nn.Linear(num_inputs, hidden_size) # First linear layer
        self.actor_linear2 = nn.Linear(hidden_size, num_actions) # Output layer (action probabilities)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (np.ndarray): The input state.

        Returns:
            tuple: A tuple containing the value estimate and the policy distribution.
        """
        # Convert state to a PyTorch Variable
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))

        # Critic forward pass
        value = F.relu(self.critic_linear1(state)) # First layer with ReLU activation
        value = self.critic_linear2(value) # Output layer (value estimate)

        # Actor forward pass
        policy_dist = F.relu(self.actor_linear1(state)) # First layer with ReLU activation
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1) # Output layer (action probabilities) with softmax

        return value, policy_dist # Return the value and policy distribution

# ----------------------------------------------------------------------
# A2C Training Function
# ----------------------------------------------------------------------
def a2c(env):
    """
    Implements the Advantage Actor-Critic (A2C) algorithm.

    Args:
        env (gym.Env): The environment to train in.
    """
    # Get environment information
    num_inputs = env.observation_space.shape[0]  # State dimension
    num_outputs = env.action_space.n # Number of actions

    # Initialize the ActorCritic network
    actor_critic = ActorCritic(num_inputs, num_outputs, HIDDEN_SIZE)

    # Initialize the Adam optimizer
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=LEARNING_RATE)

    # Lists to store training data for plotting
    all_lengths = [] # List to store episode lengths
    average_lengths = [] # List to store average episode lengths
    all_rewards = [] # List to store episode rewards

    # Training loop
    for episode in range(MAX_EPISODES):
        log_probs = []  # List to store log probabilities of taken actions
        values = [] # List to store value estimates
        rewards = [] # List to store rewards for the episode

        # Reset the environment
        state = env.reset()

        # Run the episode
        for steps in range(NUM_STEPS):
            # Get value and policy distribution from the ActorCritic network
            value, policy_dist = actor_critic.forward(state)

            # Detach the value from the computation graph
            value = value.detach().numpy()[0, 0]

            # Get the policy distribution as a NumPy array
            dist = policy_dist.detach().numpy()

            # Choose an action based on the policy distribution
            action = np.random.choice(num_outputs, p=np.squeeze(dist))

            # Calculate the log probability of the chosen action
            log_prob = torch.log(policy_dist.squeeze(0)[action])

            # Calculate the entropy of the policy distribution
            entropy = -np.sum(np.mean(dist) * np.log(dist))

            # Take a step in the environment
            new_state, reward, done, _ = env.step(action)

            # Store the reward, value, and log probability
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)

            # Update the entropy term (for exploration)
            entropy_term += entropy

            # Update the state
            state = new_state

            # If the episode is done, break the loop
            if done or steps == NUM_STEPS - 1:
                # Calculate the Q-value for the final state
                Qval, _ = actor_critic.forward(new_state)
                Qval = Qval.detach().numpy()[0, 0]

                # Store the episode reward and length
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)

                # Calculate the average length over the last 10 episodes
                average_lengths.append(np.mean(all_lengths[-10:]))

                # Print episode information every 10 episodes
                if episode % 10 == 0:
                    sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
                break

        # --------------------------------------------------------------
        # Compute Q values (Monte Carlo return)
        # --------------------------------------------------------------
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval  # Monte Carlo return
            Qvals[t] = Qval # Assign the calculated Q-value

        # --------------------------------------------------------------
        # Update actor and critic networks
        # --------------------------------------------------------------
        # Convert lists to PyTorch tensors
        values = torch.FloatTensor(values) # Values
        Qvals = torch.FloatTensor(Qvals) # Q-values
        log_probs = torch.stack(log_probs) # Log probabilities

        # Calculate advantage (A(s, a) = Q(s, a) - V(s))
        advantage = Qvals - values # Advantage function

        # Calculate actor loss (policy loss)
        actor_loss = (-log_probs * advantage).mean() # Policy gradient theorem

        # Calculate critic loss (value loss)
        critic_loss = 0.5 * advantage.pow(2).mean()  # Mean squared error between Q-values and value estimates

        # Calculate total loss (actor loss + critic loss + entropy bonus)
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term # Total loss

        # --------------------------------------------------------------
        # Optimization
        # --------------------------------------------------------------
        # Clear the optimizer gradients
        ac_optimizer.zero_grad()

        # Compute the gradients of the loss
        ac_loss.backward()

        # Update the network parameters
        ac_optimizer.step()

    # Plot results
    smoothed_rewards = pd.Series(all_rewards).rolling(10).mean() # Smooth the rewards
    plt.plot(all_rewards, label="Episode Reward")
    plt.plot(smoothed_rewards, label="Smoothed Reward (10-episode rolling mean)")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title("A2C Training - Reward")
    plt.legend()
    plt.show()

    plt.plot(all_lengths, label="Episode Length")
    plt.plot(average_lengths, label="Average Length (10-episode rolling mean)")
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.title("A2C Training - Episode Length")
    plt.legend()
    plt.show()

# ----------------------------------------------------------------------
# Main Script
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Create the CartPole environment
    env = gym.make("CartPole-v0")

    # Run the A2C algorithm
    a2c(env)
