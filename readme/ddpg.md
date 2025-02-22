# ddpg: DDPG Implementation for Continuous Actions

This project implements the Deep Deterministic Policy Gradient (DDPG) algorithm for continuous control environments, as part of a Reinforcement Learning course (TME7).

## Overview

The goal of this assignment is to implement and evaluate the DDPG algorithm, a model-free, off-policy algorithm designed for environments with continuous action spaces. DDPG combines aspects of Deep Q-Networks (DQN) and deterministic policy gradients.

## Files

*   `ddpg.py`: The main Python script containing the DDPG implementation, including the actor and critic networks, replay memory, training loop, and environment setup.

## Environment Setup

1.  **Install Dependencies:**
    ```bash
    pip install gym matplotlib numpy torch
    ```

## Algorithms Implemented

### Deep Deterministic Policy Gradient (DDPG)

*   A model-free, off-policy algorithm for continuous action spaces.
*   Uses two neural networks: an actor network to learn the optimal policy and a critic network to estimate the Q-value function.
*   Employs experience replay to decorrelate the training data and improve stability.
*   Uses target networks to stabilize the learning process by providing a fixed target for Q-value and policy updates.
*   Adds Ornstein-Uhlenbeck noise to the actions for exploration.

## Usage

1.  Run the script:
    ```bash
    python ddpg.py
    ```

2.  The script will:
    *   Create the MountainCarContinuous-v0 environment (or another continuous action environment of your choice).
    *   Train the DDPG agent.
    *   Generate a plot of the episode rewards and average rewards over time, displaying the learning progress.

## Code Structure

*   **`OUNoise` Class:** Implements the Ornstein-Uhlenbeck process for generating noise to explore the action space.
    *   `__init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000)`: Initializes the OU noise process.
    *   `reset(self)`: Resets the noise process to its initial state.
    *   `evolve_state(self)`: Evolves the state of the noise process.
    *   `get_action(self, action, t=0)`: Adds noise to the given action and clips it to the action space bounds.

*   **`NormalizedEnv` Class:** Implements an action wrapper that normalizes the actions to the range [-1, 1].
    *   `_action(self, action)`: Transforms the action to the environment's action space.
    *   `_reverse_action(self, action)`: Reverses the transformation to get the original action.

*   **`Memory` Class:** Implements a replay memory buffer for storing experiences.
    *   `__init__(self, max_size)`: Initializes the replay memory with a specified capacity.
    *   `push(self, state, action, reward, next_state, done)`: Saves a transition to memory.
    *   `sample(self, batch_size)`: Samples a random batch of transitions from memory.
    *   `__len__(self)`: Returns the number of transitions currently stored in memory.

*   **`Critic` Class:** Implements the critic network for estimating the Q-value function.
    *   `__init__(self, input_size, hidden_size, output_size=1)`: Initializes the Critic network.
    *   `forward(self, state, action)`: Performs a forward pass through the network to compute the Q-value for a given state-action pair.

*   **`Actor` Class:** Implements the actor network for generating actions.
    *   `__init__(self, input_size, hidden_size, output_size)`: Initializes the Actor network.
    *   `forward(self, state)`: Performs a forward pass through the network to compute the action for a given state.

*   **`DDPGagent` Class:** Implements the DDPG agent.
    *   `__init__(self, env, hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=50000)`: Initializes the DDPG agent.
    *   `get_action(self, state)`: Selects an action based on the current state.
    *   `update(self, batch_size)`: Updates the actor and critic networks based on a sampled batch of experiences.

## Key Parameters

*   `hidden_size`: The number of units in the hidden layers of the actor and critic networks.
*   `actor_learning_rate`: The learning rate for the actor network optimizer.
*   `critic_learning_rate`: The learning rate for the critic network optimizer.
*   `gamma`: The discount factor, which determines the importance of future rewards.
*   `tau`: The soft update coefficient, which controls the rate at which the target networks are updated.
*   `max_memory_size`: The maximum size of the replay memory buffer.

## Discussion and Further Exploration

*   **Environments:** Test the DDPG implementation on different continuous control environments, such as the Pendulum-v0 or LunarLanderContinuous-v2 environments.
*   **Hyperparameter Tuning:** Experiment with different values of the hyperparameters, such as the learning rates, hidden size, discount factor, soft update coefficient, and noise parameters, to observe their effect on the learning process.
*   **Network Architectures:** Investigate different network architectures for the actor and critic networks, such as adding more hidden layers, using different activation functions, or incorporating batch normalization.
*   **Exploration Strategies:** Explore different exploration strategies, such as using different noise processes or adaptive noise scaling.
*   **Q-Prop:** Implement the Q-Prop algorithm (as mentioned in the bonus task of the assignment) and compare its performance to DDPG.

## Notes

*   The code uses the PyTorch framework for implementing the neural networks. Make sure you have PyTorch installed.
*   This implementation provides a basic DDPG solution for continuous control environments. You can further improve the performance by tuning the hyperparameters, using more advanced network architectures, or implementing other DDPG extensions.
