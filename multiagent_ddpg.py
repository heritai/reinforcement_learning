# -*- coding: utf-8 -*-
"""multiagent_ddpg.py

This script implements the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm for multi-agent environments, as part of a Reinforcement Learning course.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import multiagent
import numpy as np
from copy import deepcopy
from collections import namedtuple
import random
import matplotlib.pyplot as plt
import warnings
import sys
import traceback

#-----------------------------------------------------------------------
# Configuration
#-----------------------------------------------------------------------
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    """
    Custom warning handler that includes traceback information.
    """
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback
warnings.simplefilter("error")  # Treat warnings as errors

#-----------------------------------------------------------------------
# Helper Functions
#-----------------------------------------------------------------------
def make_env(scenario_name, benchmark=False):
    """
    Creates a multi-agent environment.

    Args:
        scenario_name (str): The name of the scenario to load.
        benchmark (bool): Whether to use the benchmark environment.

    Returns:
        MultiAgentEnv: The multi-agent environment.
    """
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

#-----------------------------------------------------------------------
# Network Definitions
#-----------------------------------------------------------------------
class CriticNet(nn.Module):
    """
    Critic network for estimating Q-values in a multi-agent setting.
    """
    def __init__(self, dim_observation, dim_action):
        """
        Initializes the Critic network.

        Args:
            dim_observation (int): The number of dimensions in the state space.
            dim_action (int): The number of dimensions in the action space.
        """
        super(CriticNet, self).__init__()

        obs_dim = dim_observation
        act_dim = dim_action

        self.FC1 = nn.Linear(obs_dim, 1024)
        self.FC2 = nn.Linear(1024+act_dim, 512)
        self.FC3 = nn.Linear(512, 300)
        self.FC4 = nn.Linear(300, 1)

    def forward(self, obs, acts):
        """
        Forward pass through the network.

        Args:
            obs (torch.Tensor): The input state (batch_size x obs_dim).
            acts (torch.Tensor): The input actions (batch_size x act_dim).

        Returns:
            torch.Tensor: The estimated Q-values (batch_size x 1).
        """
        result = torch.relu(self.FC1(obs))
        combined = torch.cat([result, acts], 1)
        out = torch.relu(self.FC2(combined))
        return self.FC4(torch.relu(self.FC3(out)))

class ActorNet(nn.Module):
    """
    Actor network for generating actions in a multi-agent setting.
    """
    def __init__(self, dim_observation, dim_action):
        """
        Initializes the Actor network.

        Args:
            dim_observation (int): The number of dimensions in the state space.
            dim_action (int): The number of dimensions in the action space.
        """
        super(ActorNet, self).__init__()

        self.FC1 = nn.Linear(dim_observation, 500)
        self.FC2 = nn.Linear(500, 128)
        self.FC3 = nn.Linear(128, dim_action)

    def forward(self, obs):
        """
        Forward pass through the network.

        Args:
            obs (torch.Tensor): The input state (batch_size x obs_dim).

        Returns:
            torch.Tensor: The generated actions (batch_size x act_dim), scaled to be between -1 and 1.
        """
        result = torch.relu(self.FC1(obs))
        result = torch.relu(self.FC2(result))
        result = torch.tanh(self.FC3(result))
        return result

#-----------------------------------------------------------------------
# Replay Buffer
#-----------------------------------------------------------------------
Experience = namedtuple('Experience', ('states', 'actions', 'next_states', 'rewards', 'dones'))

class ReplayMemory:
    """
    Replay memory for storing experiences.
    """
    def __init__(self, capacity):
        """
        Initializes the ReplayMemory.

        Args:
            capacity (int): The maximum number of experiences to store.
        """
        self.capacity = capacity # Maximum capacity
        self.memory = [] # List to store experiences
        self.position = 0 # Current position in the memory

    def push(self, *args):
        """
        Adds an experience to the replay memory.

        Args:
            *args: The experience tuple (states, actions, next_states, rewards, dones).
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None) # Append a placeholder if memory is not full
        self.memory[self.position] = Experience(*args) # Store the experience
        self.position = (self.position + 1) % self.capacity # Circular buffer

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the replay memory.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            list: A list of sampled experiences.
        """
        return random.sample(self.memory, batch_size) # Sample a random batch

    def __len__(self):
        """
        Returns the number of experiences in the replay memory.
        """
        return len(self.memory) # Return the length of the memory

#-----------------------------------------------------------------------
# MADDPG Agent
#-----------------------------------------------------------------------
class MADDPG_Agent:
    """
    Centralized Training, Decentralized Execution Multi-Agent DDPG.
    """
    def __init__(self, env, batch_size, replay_capacity, episodes_before_train, device='cpu'):
        """
        Initializes the MADDPG agent.

        Args:
            env (MultiAgentEnv): The multi-agent environment.
            batch_size (int): The batch size for training.
            replay_capacity (int): The capacity of the replay memory.
            episodes_before_train (int): The number of episodes to run before starting training.
            device (str): The device to run the training on ('cpu' or 'cuda').
        """
        self.env = env # Multi-agent environment
        self.n_agents = env.n # Number of agents
        self.memory = ReplayMemory(replay_capacity) # Replay memory

        # Initialize actor and critic networks for each agent
        self.actors = [ActorNet(env.observation_space[i].shape[0], env.action_space[i].shape[0]) for i in range(self.n_agents)] # Actor networks
        self.critics = [CriticNet(sum([env.observation_space[i].shape[0] for i in range(self.n_agents)]), sum([env.action_space[i].shape[0] for i in range(self.n_agents)])) for i in range(self.n_agents)] # Critic networks (centralized)


        # Initialize optimizers for actor and critic networks
        self.critic_optimizers = [optim.Adam(x.parameters(), lr=0.01) for x in self.critics] # Critic optimizers
        self.actor_optimizers = [optim.Adam(x.parameters(), lr=0.01) for x in self.actors] # Actor optimizers

        # Initialize target networks for actor and critic networks
        self.actor_targets = deepcopy(self.actors) # Target actor networks
        self.critic_targets = deepcopy(self.critics) # Target critic networks

        self.device = device # Device to run the training on
        self.episodes_before_train = episodes_before_train # Number of episodes before training starts
        self.batch_size = batch_size # Batch size for training

        # Hyperparameters
        self.GAMMA = 0.95 # Discount factor
        self.epsilon = 0.3 # Exploration rate
        self.rewards_list = [] # List to store rewards for each episode

        # Move networks to the specified device
        for x in self.actors:           x.to(device)
        for x in self.critics:          x.to(device)
        for x in self.actor_targets:    x.to(device)
        for x in self.critic_targets:   x.to(device)

    def select_actions(self, actor_nets, states, noise=True):
        """
        Selects actions for each agent based on their individual policies.

        Args:
            actor_nets (list): A list of actor networks.
            states (np.ndarray): The current states of all agents.
            noise (bool): Whether to add exploration noise to the actions.

        Returns:
            list: A list of actions for each agent.
        """
        actions = [] # List to store the actions
        for actor, state in zip(actor_nets, states): # Iterate over the actor networks and states
            state_v = torch.from_numpy(state).float().to(self.device) # Convert the state to a PyTorch tensor

            actor.eval()  # Set the actor network to evaluation mode for inference
            with torch.no_grad():
              action = actor(state_v).cpu().numpy()  # Get the action from the actor network


            actor.train() # Set the actor network to training mode

            if noise:
                action += self.epsilon * np.random.normal(size=action.shape) # Add exploration noise
            action = np.clip(action, -1, 1) # Clip the action to the range [-1, 1]
            actions.append(action) # Append the action to the list

        return actions # Return the list of actions

    def learn(self, batch):
        """
        Updates the actor and critic networks for all agents based on a sampled batch of experiences.

        Args:
            batch (Experience): A batch of experiences sampled from the replay memory.
        """
        states_batch, actions_batch, next_states_batch, rewards_batch, dones_batch = batch # Unpack the batch

        # Convert data to PyTorch tensors
        rewards_batch = torch.tensor(np.array(rewards_batch), dtype=torch.float, device=self.device)
        dones_batch = torch.tensor(np.array(dones_batch), dtype=torch.bool, device=self.device)
        next_states_batch = [torch.tensor(s, dtype=torch.float, device=self.device) for s in next_states_batch]
        states_batch = [torch.tensor(s, dtype=torch.float, device=self.device) for s in states_batch]
        actions_batch = [torch.tensor(a, dtype=torch.float, device=self.device) for a in actions_batch]


        all_agents = np.arange(self.n_agents)
        c_loss, a_loss = [], []
        # Iterate over each agent to update their actor and critic networks
        for agent in range(self.n_agents):
            #-----------------------------------------------------------------------
            # Critic Update
            #-----------------------------------------------------------------------
            self.critic_optimizers[agent].zero_grad() # Zero the gradients of the critic optimizer

            # Get current and next actions for all agents
            critic_net_inps = torch.cat(states_batch, dim=1)
            target_actions = [self.actor_targets[i](next_states_batch[i]) for i in all_agents]
            target_actions = torch.cat(target_actions, dim=1)
            with torch.no_grad():
                target_value = self.critic_targets[agent](torch.cat(next_states_batch, dim=1), target_actions) # target critic Q value

            # Compute the target Q value
            target_Q = rewards_batch[:, agent].unsqueeze(1) + self.GAMMA * target_value * (~dones_batch[:, agent].unsqueeze(1))  # reward + gamma * Q(s', a')

            # Compute the critic loss
            estimated_Q = self.critics[agent](critic_net_inps, torch.cat(actions_batch, dim=1))
            critic_loss = F.mse_loss(estimated_Q, target_Q)

            # Update the critic network
            critic_loss.backward() # Compute the gradients of the loss
            self.critic_optimizers[agent].step() # Update the critic network parameters

            c_loss.append(critic_loss.item()) # Store critic loss for logging

            #-----------------------------------------------------------------------
            # Actor Update
            #-----------------------------------------------------------------------
            self.actor_optimizers[agent].zero_grad() # Zero the gradients of the actor optimizer

            # Compute the actor loss
            actor_actions = [self.actors[i](states_batch[i]) if i == agent else actions_batch[i] for i in all_agents] # Get the actions from all actor networks
            actor_actions = torch.cat(actor_actions, dim=1) # Concatenate all actions

            # Calculate the actor loss
            actor_loss = -self.critics[agent](critic_net_inps, actor_actions).mean() # Calculate the actor loss
            actor_loss.backward() # Compute the gradients of the loss
            self.actor_optimizers[agent].step() # Update the actor network parameters

            a_loss.append(actor_loss.item()) # Store actor loss for logging


        return (c_loss, a_loss)

    def soft_update(self, target, source, t):
        """
        Softly updates the parameters of the target network using the parameters of the source network.

        Args:
            target (nn.Module): The target network.
            source (nn.Module): The source network.
            t (float): The soft update coefficient.
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data) # Soft update

    def train(self, n_episodes, max_episode_length=100):
        """
        Trains the MADDPG agent.

        Args:
            n_episodes (int): The number of episodes to train for.
            max_episode_length (int): The maximum length of each episode.
        """
        total_steps = 0 # Total number of steps taken
        for i_episode in range(n_episodes): # Iterate over each episode

            states = env.reset() # Reset the environment
            episode_rewards = []
            steps = 0 # Number of steps taken in the current episode

            # Run the episode
            for t in range(max_episode_length):
                # Select actions for all agents
                actions = self.select_actions(self.actors, states, noise=True)

                # Take a step in the environment
                next_states, rewards, dones, infos = env.step(actions)
                episode_rewards.append(sum(rewards))

                dones = [dones[i] for i in range(self.n_agents)]

                # Store the experience in the replay memory
                self.memory.push(states, actions, next_states, rewards, dones)

                # Update the current state
                states = next_states

                # Learn from the experiences in the replay memory
                if i_episode > self.episodes_before_train and len(self.memory) >= self.batch_size:
                    c_losses, a_losses = self.learn(self.memory.sample(self.batch_size))

                    # Update the target networks using soft updates
                    for i in range(self.n_agents):
                        self.soft_update(self.actor_targets[i], self.actors[i], t=0.01)
                        self.soft_update(self.critic_targets[i], self.critics[i], t=0.01)

                # Check if the episode is done
                if all(dones) or t >= max_episode_length - 1:
                    break

                steps += 1  # Increment the number of steps
                total_steps += 1 # Increment the total number of steps


            print(f'Episode: {i_episode}, reward = {sum(episode_rewards):.3f}')
            self.rewards_list.append(sum(episode_rewards))

#-----------------------------------------------------------------------
# Main Execution
#-----------------------------------------------------------------------
if __name__ == '__main__':
    device = 'cpu'  # Set the device to 'cpu' or 'cuda'
    # Create the multi-agent environment
    #env = make_env('simple_tag')
    env = make_env('simple_spread')

    # Create the MADDPG agent
    agent = MADDPG_Agent(env, 64, 10000, 10, device=device)

    # Train the agent
    agent.train(400, 200)

    # Plot the rewards
    import matplotlib.pyplot as plt
    plt.plot(agent.rewards_list)
    plt.xlabel("epochs")
    plt.ylabel("reward")
    plt.show()
