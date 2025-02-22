# -*- coding: utf-8 -*-
"""qlearning_dynaq.py

This script implements Q-learning, SARSA, and Dyna-Q algorithms for solving
the GridWorld environment, as part of a Reinforcement Learning course.
"""

import gym
import gridworld  # Assuming gridworld.py is in the same directory
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib for plotting


# ----------------------------------------------------------------------
# Base Agent Class (Abstract)
# ----------------------------------------------------------------------
class BaseAgent(object):
    """
    Abstract base class for reinforcement learning agents.
    """
    def __init__(self, states, actions, lr=0.01, gamma=1, eps=0.3):
        """
        Initializes the agent.

        Args:
            states (list): List of all possible states.
            actions (gym.spaces.Discrete): The action space.
            lr (float): Learning rate (alpha).
            gamma (float): Discount factor.
            eps (float): Exploration rate (epsilon) for epsilon-greedy policy.
        """
        self.states = states
        self.actions = np.arange(actions.n) # All possible actions
        self.lr = lr
        self.eps = eps
        self.gamma = gamma
        self.qtable = self._initialize_q_table() # Initialize the Q-table

    def _initialize_q_table(self):
        """
        Initializes the Q-table with zeros for all state-action pairs.

        Returns:
            dict: A dictionary representing the Q-table (state -> {action -> Q-value}).
        """
        qtable = {}
        for s in self.states:
            qtable[s] = {a: 0 for a in self.actions} # Initialize Q-values to 0
        return qtable

    def epsGreedpolicy(self, obs):
        """
        Epsilon-greedy policy for action selection.

        Args:
            obs (str): The current state (observation).

        Returns:
            int: The action to take.
        """
        if np.random.rand() > self.eps:
            # Exploit: Choose the action with the highest Q-value
            a = max(self.qtable[obs], key=self.qtable[obs].get) # Get the action with the highest Q-value for the current state
        else:
            # Explore: Choose a random action
            a = np.random.choice(list(self.qtable[obs].keys()))  # Randomly select action
        return a

    def act(self, obs):
        """
        Selects an action based on the current state and the epsilon-greedy policy.

        Args:
            obs (np.ndarray): The current observation from the environment.

        Returns:
            int: The action to take.
        """
        obs_str = str(obs.tolist()) # Convert observation to string for dictionary lookup
        action = self.epsGreedpolicy(obs_str)  # Use epsilon-greedy policy to select action
        return action

    def updateQvals(self, *args, **kwargs):
        """
        Abstract method for updating Q-values.  Must be implemented by subclasses.
        """
        raise NotImplementedError

# ----------------------------------------------------------------------
# Q-learning Agent Class
# ----------------------------------------------------------------------
class QlearningAgent(BaseAgent):
    """
    An agent that learns using the Q-learning algorithm.
    """
    def __init__(self, states, actions, lr=0.01, gamma=1, eps=0.3):
        super().__init__(states, actions, lr, gamma, eps) # Call the parent's constructor

    def updateQvals(self, st0, st1, action, r, done):
        """
        Updates the Q-value for the given state-action pair using the Q-learning update rule.

        Args:
            st0 (np.ndarray): The initial state.
            st1 (np.ndarray): The next state.
            action (int): The action taken.
            r (float): The reward received.
            done (bool): Whether the episode is done.
        """
        st0_str = str(st0.tolist()) # Convert states to strings for Q-table lookup
        st1_str = str(st1.tolist())

        # Q-learning update rule:
        # Q(s, a) = Q(s, a) + lr * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
        best_next_q = max(self.qtable[st1_str].values()) # Get the maximum Q-value for the next state
        current_q = self.qtable[st0_str][action] # Get the current Q-value

        # Update the Q-value
        self.qtable[st0_str][action] = current_q + self.lr * (r + self.gamma * best_next_q - current_q)

# ----------------------------------------------------------------------
# SARSA Agent Class
# ----------------------------------------------------------------------
class SARSAAgent(BaseAgent):
    """
    An agent that learns using the SARSA algorithm.
    """
    def __init__(self, states, actions, lr=0.01, gamma=1, eps=0.3):
        super().__init__(states, actions, lr, gamma, eps)  # Call the parent's constructor

    def updateQvals(self, st0, st1, action1, action2, r, done):
        """
        Updates the Q-value for the given state-action pair using the SARSA update rule.

        Args:
            st0 (np.ndarray): The initial state.
            st1 (np.ndarray): The next state.
            action1 (int): The action taken in the initial state.
            action2 (int): The action taken in the next state.
            r (float): The reward received.
            done (bool): Whether the episode is done.
        """
        st0_str = str(st0.tolist()) # Convert states to strings for Q-table lookup
        st1_str = str(st1.tolist())

        # SARSA update rule:
        # Q(s, a) = Q(s, a) + lr * [reward + gamma * Q(s', a') - Q(s, a)]
        current_q = self.qtable[st0_str][action1] # Get the current Q-value
        next_q = self.qtable[st1_str][action2]  # Get the Q-value for the next state-action pair

        # Update the Q-value
        self.qtable[st0_str][action1] = current_q + self.lr * (r + self.gamma * next_q - current_q)


# ----------------------------------------------------------------------
# Dyna-Q Agent Class
# ----------------------------------------------------------------------
class DynaQAgent(BaseAgent):
    """
    An agent that learns using the Dyna-Q algorithm.
    """
    def __init__(self, states, actions, terminal_states, lr=0.01, gamma=1, eps=0.3, k=10, alphaUpdate=0.01):
        """
        Initializes the Dyna-Q agent.

        Args:
            states (list): List of all possible states.
            actions (gym.spaces.Discrete): The action space.
            terminal_states (list): List of terminal states.
            lr (float): Learning rate.
            gamma (float): Discount factor.
            eps (float): Exploration rate.
            k (int): Number of planning steps (updates from the model).
            alphaUpdate (float): Learning rate for updating the model.
        """
        super().__init__(states, actions, lr, gamma, eps) # Call the parent's constructor
        self.terminals = terminal_states
        self.alphaUpdate = alphaUpdate
        self.k = k
        self.model = self._initialize_model() # Initialize the model

    def _initialize_model(self):
        """
        Initializes the model. The model is a dictionary that stores the predicted reward and next state for each state-action pair.

        Returns:
            dict: A dictionary representing the model (state -> {action -> {next_state -> [reward, probability]}}).
        """
        model = {}
        for s1 in self.states:
            model[s1] = {}
            for a in self.actions:
                model[s1][a] = {}
                for s2 in self.states:
                    model[s1][a][s2] = [0, 0]  # [reward, probability]
        return model

    def updateModel(self, st0, st1, action, r, alphaR=0.1):
        """
        Updates the model based on the observed transition.

        Args:
            st0 (str): The initial state (as a string).
            st1 (str): The next state (as a string).
            action (int): The action taken.
            r (float): The reward received.
            alphaR (float): Learning rate for updating the model.
        """
        rr = self.model[st0][action][st1][0]
        self.model[st0][action][st1][0] = rr + alphaR * (r - rr) # Update the reward estimate
        pp = self.model[st0][action][st1][1]
        self.model[st0][action][st1][1] = pp + alphaR * (1 - pp) # Update the probability estimate
        for ss in self.states:
            if ss != st1:
                pp = self.model[st0][action][ss][1]
                self.model[st0][action][ss][1] = pp + alphaR * (0 - pp) # Decrease the probability for other states

    def updatebyModel(self, alphaUpdate, k):
        """
        Performs planning steps by updating the Q-values based on the model.

        Args:
            alphaUpdate (float): Learning rate for updating the Q-values.
            k (int): Number of planning steps to perform.
        """
        statesToUpdate = np.random.choice(list(self.states.keys()), k) # Select random states to update
        actionsToUpdate = np.random.choice(self.actions, k) # Select random actions to update

        for s, a in zip(statesToUpdate, actionsToUpdate):
            if s not in self.terminals:
                ssdict = self.model[s][a] # Get the model predictions for the state-action pair
                qq = self.qtable[s][a] # Get the current Q-value

                # Update the Q-value based on the model predictions
                self.qtable[s][a] = qq + alphaUpdate * (np.sum([ssdict[s1][1] * (ssdict[s1][0] + self.gamma * max(self.qtable[s1].values())) for s1 in ssdict]) - qq)

    def updateQvals(self, st0, st1, action, r, done):
        """
        Updates the Q-value and the model.

        Args:
            st0 (np.ndarray): The initial state.
            st1 (np.ndarray): The next state.
            action (int): The action taken.
            r (float): The reward received.
            done (bool): Whether the episode is done.
        """
        st0_str = str(st0.tolist()) # Convert states to strings for Q-table lookup
        st1_str = str(st1.tolist())

        # Q-learning update rule:
        # Q(s, a) = Q(s, a) + lr * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
        best_next_q = max(self.qtable[st1_str].values()) # Get the maximum Q-value for the next state
        current_q = self.qtable[st0_str][action] # Get the current Q-value

        # Update the Q-value
        self.qtable[st0_str][action] = current_q + self.lr * (r + self.gamma * best_next_q - current_q)

        self.updateModel(st0_str, st1_str, action, r)  # Update the model
        self.updatebyModel(self.alphaUpdate, self.k)  # Update Q-values based on the model


# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
# Create the GridWorld environment
env = gym.make("gridworld-v0")
env.seed(0)  # Initialize the random seed

# Set the environment's plan (grid layout and rewards)
env.setPlan("gridworldPlans/plan4.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

# Get the state dictionary and terminal states
statedic, mdp = env.getMDP()
terminal_states = list(set(statedic.keys()).difference(set(mdp.keys()))) # Get the terminal states

# Initialize the agents
qlAgent = QlearningAgent(statedic, env.action_space, lr=0.1, gamma=0.9, eps=0.4)
sarsaAgnt = SARSAAgent(statedic, env.action_space, lr=0.1, gamma=0.9, eps=0.4)
dynQAgnt = DynaQAgent(statedic, env.action_space, terminal_states, lr=0.1, gamma=0.9, eps=0.4, k=100, alphaUpdate=0.4)


# ----------------------------------------------------------------------
# Training Loop
# ----------------------------------------------------------------------
def train_agent(agent, env, episode_count, agent_name):
    """
    Trains the given agent in the given environment.

    Args:
        agent (BaseAgent): The agent to train.
        env (gym.Env): The environment to train in.
        episode_count (int): The number of episodes to train for.
        agent_name (str): The name of the agent.

    Returns:
        list: A list of cumulative rewards for each episode.
    """
    cumulative_rewards = [] # Store cumulative rewards for each episode

    for i in range(episode_count):
        obs = env.reset() # Reset the environment at the beginning of each episode
        rsum = 0 # Initialize cumulative reward for the episode
        j = 0 # Initialize step counter for the episode

        if agent_name == "SARSA":
            action1 = agent.act(obs)  # SARSA: Choose the first action
        else:
            action1=None

        while True:
            if agent_name != "SARSA":
              action = agent.act(obs) # Choose action based on the agent's policy
            else:
              action=action1

            st0 = obs.copy() # Store the current state

            obs, reward, done, _ = env.step(action) # Take a step in the environment

            if agent_name == "SARSA":
                action2 = agent.act(obs) # SARSA: Choose the next action
                agent.updateQvals(st0, obs, action, action2, reward, done)  # SARSA: Update Q-values with the next action
                action1=action2
            else:
                agent.updateQvals(st0, obs, action, reward, done) # Update Q-values

            rsum += reward # Accumulate the reward
            j += 1 # Increment the step counter

            if done:
                cumulative_rewards.append(rsum)  # Store the cumulative reward
                if i % 100 == 0:
                    print(f"{agent_name} - Episode: {i}, rsum={rsum}, actions={j}")
                break

    return cumulative_rewards

# Train the agents
episode_count = 10000
rqAgent = train_agent(qlAgent, env, episode_count, "Q-learning")

episode_count = 10000
rSarsAgent = train_agent(sarsaAgnt, env, episode_count, "SARSA")

episode_count = 1000
rDyna = train_agent(dynQAgnt, env, episode_count, "Dyna-Q")



# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
# Plotting the cumulative rewards for all agents
plt.figure(figsize=(12, 6))  # Adjust figure size for better visualization

plt.plot(np.cumsum(rqAgent), label="Q-learning")
plt.plot(np.cumsum(rSarsAgent), label="SARSA")
plt.plot(np.cumsum(rDyna), label="Dyna-Q")

plt.xlabel("Episode")
plt.ylabel("Cumulative reward")
plt.title("Cumulative Reward per Episode")
plt.legend()  # Show the legend
plt.savefig("cumulative_reward_comparison.png")  # Save the plot
plt.show()

print("done")
env.close()
