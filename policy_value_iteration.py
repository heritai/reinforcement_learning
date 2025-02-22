# -*- coding: utf-8 -*-
"""policy_value_iteration.py

This script implements Policy Iteration and Value Iteration algorithms for solving
the GridWorld environment, as part of a Reinforcement Learning course.
"""

import os
from google.colab import drive
import matplotlib.pyplot as plt  # Import pyplot for plotting

#matplotlib.use("TkAgg") #Commented out to avoid TkAgg dependency issues in environments without a display.
import gym
from gym import wrappers
import numpy as np
import copy

import gridworld  # Import the custom GridWorld environment

# ----------------------------------------------------------------------
# Policy Iteration Agent
# ----------------------------------------------------------------------
class PolicyIterAgent(object):
  """
  An agent that learns an optimal policy using Policy Iteration.
  """
  def __init__ (self, env, gamma=1, eps=0.0001, exploration_rate=0.1):
      """
      Initializes the Policy Iteration agent.

      Args:
          env: The GridWorld environment.
          gamma (float): Discount factor (default: 1).
          eps (float): Convergence threshold for policy evaluation (default: 0.0001).
          exploration_rate (float): Probability of taking a random action (default: 0.1).
      """
      self.action_space = env.action_space
      self.gamma = gamma
      self.eps = eps
      self.statedic, self.mdp = env.getMDP()  # Get state dictionary and MDP representation from the environment
      self.policyIndx = 0
      self.policy = {}
      self.exploration_rate = exploration_rate

      # Initialize policy randomly for all states
      for s in self.mdp:
        self.policy[s] = self.action_space.sample()

  def policyEval(self, pol):
    """
    Evaluates the given policy using iterative policy evaluation.

    Args:
        pol (dict): A dictionary representing the policy to evaluate (state -> action).

    Returns:
        dict: A dictionary representing the value function for the given policy (state -> value).
    """
    # Initialize value function
    V0 = {s: 0 for s in self.statedic}
    V1 = {s: 0 for s in self.statedic}

    while True:
      for s1 in self.mdp:
        # Calculate the value of state s1 under the current policy
        V1[s1] = np.sum([t[0] * (t[2] + self.gamma * V0[t[1]]) for t in self.mdp[s1][pol[s1]]])

      # Check for convergence
      if np.linalg.norm(np.array(list(V0.values())) - np.array(list(V1.values()))) < self.eps:
        break
      V0 = V1.copy()  # Use copy to avoid modifying V0 directly
    return V1

  def nextPolicy(self):
    """
    Improves the policy by selecting the best action for each state based on the current value function.
    """
    self.policyIndx += 1
    Vpi = self.policyEval(self.policy)  # Evaluate the current policy

    for s1 in self.mdp:
      # Calculate the Q-values for each action in state s1
      q_values = [np.sum([t[0] * (t[2] + self.gamma * Vpi[t[1]]) for t in self.mdp[s1][a]]) for a in range(self.action_space.n)]

      # Exploration vs. Exploitation:  Epsilon-greedy action selection
      if np.random.rand() > self.exploration_rate:
          # Exploit: Choose the action with the highest Q-value
          self.policy[s1] = np.argmax(q_values)
      else:
          # Explore: Choose a random action
          self.policy[s1] = self.action_space.sample()


  def act(self, observation):
    """
    Returns the action to take in the given observation based on the current policy.

    Args:
        observation: The current state of the environment.

    Returns:
        int: The action to take.
    """
    obs = str(observation.tolist())  # Convert observation to string for dictionary lookup
    action = self.policy[obs]
    return action

# ----------------------------------------------------------------------
# Value Iteration Agent
# ----------------------------------------------------------------------
class ValueIterAgent(object):
  """
  An agent that learns an optimal policy using Value Iteration.
  """
  def __init__ (self, env, gamma=1, eps=0.0001):
      """
      Initializes the Value Iteration agent.

      Args:
          env: The GridWorld environment.
          gamma (float): Discount factor (default: 1).
          eps (float): Convergence threshold for value iteration (default: 0.0001).
      """
      self.action_space = env.action_space
      self.gamma = gamma
      self.eps = eps
      self.statedic, self.mdp = env.getMDP()  # Get state dictionary and MDP representation
      self.valueIndx = 0
      self.isConverged = False
      self.policy = {}

      # Initialize policy randomly (will be updated during value iteration)
      for s in self.mdp:
        self.policy[s] = self.action_space.sample()

      # Initialize value function to zero for all states
      self.Val = {s: 0 for s in self.statedic}

  def valueEval(self):
    """
    Performs one iteration of value iteration, updating the value function for all states.
    """
    for s1 in self.mdp:
      # Calculate the Q-values for each action in state s1
      q_values = [np.sum([t[0] * (t[2] + self.gamma * self.Val[t[1]]) for t in self.mdp[s1][a]]) for a in range(self.action_space.n)]
      self.Val[s1] = max(q_values)  # Update the value of state s1 to the maximum Q-value

  def nextPolicy(self):
    """
    Extracts the optimal policy based on the converged value function and checks for convergence.
    """
    self.valueIndx += 1
    v1 = list(self.Val.values())  # Store the previous value function

    self.valueEval()  # Perform one iteration of value iteration

    for s1 in self.mdp:
      # Calculate the Q-values for each action in state s1
      q_values = [np.sum([t[0] * (t[2] + self.gamma * self.Val[t[1]]) for t in self.mdp[s1][a]]) for a in range(self.action_space.n)]

      self.policy[s1] = np.argmax(q_values)  # Extract the optimal action (policy) for state s1

    # Check for convergence
    self.isConverged = np.linalg.norm(np.array(list(self.Val.values())) - np.array(v1)) < self.eps


  def act(self, observation):
    """
    Returns the action to take in the given observation based on the current policy.

    Args:
        observation: The current state of the environment.

    Returns:
        int: The action to take.
    """
    obs = str(observation.tolist())  # Convert observation to string for dictionary lookup
    action = self.policy[obs]
    return action

# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
# Create the GridWorld environment
env = gym.make("gridworld-v0")
env.seed(0)  # Initialize the random seed

# Set the environment's plan (grid layout and rewards)
env.setPlan("gridworldPlans/plan2.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

# Initialize Policy Iteration and Value Iteration agents
polIterAgent = PolicyIterAgent(env, gamma=0.9, eps=1e-6)
valIterAgent = ValueIterAgent(env, gamma=0.5, eps=1e-8)

# ----------------------------------------------------------------------
# Value Iteration Training Loop
# ----------------------------------------------------------------------
rsumVec = []  # Store cumulative rewards for each episode
episode_count = 100  # Number of episodes to train for

for i in range(episode_count):
  obs = env.reset()  # Reset the environment at the beginning of each episode
  rsum = 0
  j = 0

  while True:
    action = valIterAgent.act(obs)  # Choose action based on the agent's policy
    obs, reward, done, _ = env.step(action)  # Take a step in the environment
    rsum += reward  # Accumulate the reward
    j += 1

    if done:
      print("Value Iteration - Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
      rsumVec.append(rsum)
      valIterAgent.nextPolicy()  # Improve the value function and policy
      break

# Plot the cumulative rewards for Value Iteration
plt.figure(figsize=(10, 5))  # Adjust figure size for better visualization
plt.plot(np.cumsum(rsumVec))
plt.xlabel("Episode")
plt.ylabel("Cumulative reward")
plt.title("Value Iteration - Cumulative Reward per Episode")
plt.savefig("value_iteration_cumulative_reward.png")  # Save the plot
plt.show() #show the plot

# ----------------------------------------------------------------------
# Policy Iteration Training Loop
# ----------------------------------------------------------------------
rsumvecpoliter = []  # Store cumulative rewards for each episode
episode_count = 1000  # Number of episodes to train for

for i in range(episode_count):
  obs = env.reset()  # Reset the environment at the beginning of each episode
  rsum = 0
  j = 0

  while True:
    action = polIterAgent.act(obs)  # Choose action based on the agent's policy
    obs, reward, done, _ = env.step(action)  # Take a step in the environment
    rsum += reward  # Accumulate the reward
    j += 1

    if done:
      print("Policy Iteration - Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
      polIterAgent.nextPolicy()  # Improve the policy
      rsumvecpoliter.append(rsum)
      break

# Plot the cumulative rewards for Policy Iteration
plt.figure(figsize=(10, 5))  # Adjust figure size for better visualization
plt.plot(np.cumsum(rsumvecpoliter))
plt.xlabel("Episode")
plt.ylabel("Cumulative reward")
plt.title("Policy Iteration - Cumulative Reward per Episode")
plt.savefig("policy_iteration_cumulative_reward.png")  # Save the plot
plt.show() #show the plot

print("done")
env.close()
