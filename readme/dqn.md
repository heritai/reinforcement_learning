# dqn: DQN Implementation for CartPole, GridWorld, and LunarLander

This project implements the Deep Q-Network (DQN) algorithm for solving three different reinforcement learning environments: CartPole, GridWorld, and LunarLander. It was created as part of a Reinforcement Learning course.

## Overview

The goal of this assignment is to implement and evaluate the DQN algorithm on three distinct environments, demonstrating its versatility and ability to learn optimal policies.

## Files

*   `dqn.py`: The main Python script containing the DQN implementation, including the network architecture, replay memory, training loop, and environment setups.
*   `gridworld.py`: (Assumed to be present) A custom environment for GridWorld.
*   `gridworldPlans/`: Contains various grid layouts (`planX.txt`) defining the environment.

## Environment Setup

1.  **Install Dependencies:**
    ```bash
    pip install gym matplotlib numpy torch
    ```
2.  **GridWorld Environment:** Ensure that the `gridworld.py` file and the `gridworldPlans/` directory are in the same directory as `dqn.py`. These files define the GridWorld environment.

## Algorithms Implemented

### Deep Q-Network (DQN)

*   A model-free, off-policy reinforcement learning algorithm.
*   Uses a deep neural network to approximate the Q-function, which estimates the expected cumulative reward for taking a specific action in a given state.
*   Employs experience replay to decorrelate the training data and improve stability.
*   Uses a target network to stabilize the learning process by providing a fixed target for Q-value updates.

## Usage

1.  Run the script:
    ```bash
    python dqn.py
    ```

2.  The script will:
    *   Train a DQN agent for each of the three environments (CartPole, GridWorld, and LunarLander).
    *   Print the cumulative reward for each episode during training.
    *   Generate a plot of the cumulative rewards over episodes for each environment, displaying the learning progress.

## Code Structure

*   **`Transition` Namedtuple:** Represents a single transition (state, action, next\_state, reward) for storing experience replay data.

*   **`ReplayMemory` Class:** Implements a replay memory buffer for storing and sampling past experiences.
    *   `__init__(self, capacity)`: Initializes the replay memory with a specified capacity.
    *   `push(self, *args)`: Saves a transition to memory.
    *   `sample(self, batch_size)`: Samples a random batch of transitions from memory.
    *   `__len__(self)`: Returns the number of transitions currently stored in memory.

*   **`DQN` Class:** Implements the Deep Q-Network architecture.
    *   `__init__(self, inSize, outSize, layers=[])`: Initializes the DQN with a specified input size, output size, and hidden layer configuration.
    *   `forward(self, x)`: Performs a forward pass through the network to compute Q-values for a given state.

*   **`FeaturesExtractor` Class:** Implements a feature extractor for the GridWorld environment.
    *   `__init__(self, outSize)`: Initializes the feature extractor with a specified output size.
    *   `getFeatures(self, obs)`: Extracts features from the raw observation, representing the positions of the agent, yellow elements, and rose elements.

*   **`train` Function:** Implements the main training loop for the DQN agent.
    *   Takes the environment, number of episodes, hidden layer configuration, and feature extractor (optional) as input.
    *   Initializes the policy network, target network, optimizer, and replay memory.
    *   Implements the epsilon-greedy action selection strategy.
    *   Performs Q-value updates using the Bellman equation and Huber loss.
    *   Updates the target network periodically to stabilize learning.

*   **`plot_results` Function:** Plots the cumulative rewards over episodes to visualize the learning progress.

## Key Parameters

*   `BATCH_SIZE`: The number of transitions sampled from the replay memory for each training update.
*   `GAMMA`: The discount factor, which determines the importance of future rewards.
*   `EPS_START`: The initial exploration rate for the epsilon-greedy action selection strategy.
*   `EPS_END`: The final exploration rate for the epsilon-greedy action selection strategy.
*   `EPS_DECAY`: The decay rate for the exploration rate.
*   `TARGET_UPDATE`: The frequency at which the target network is updated.
*   `hidden_layers`:  List of hidden layer sizes for the DQN.

## Environment Specifics

### 1. CartPole

*   Observation Space: 4-dimensional vector representing the position and velocity of the cart and pole.
*   Action Space: Discrete actions (0: push cart to the left, 1: push cart to the right).
*   Reward: +1 for every step taken, including the termination step.
*   Termination: Episode ends if the pole falls more than 15 degrees from vertical or the cart moves more than 2.4 units from the center.

### 2. GridWorld

*   Observation Space: Grid-based representation of the environment, including the agent, yellow elements, and rose elements.
*   Action Space: Discrete actions (0: up, 1: right, 2: down, 3: left).
*   Reward: Defined by the `setPlan` method, typically -0.001 for empty cells, +1 for green and yellow cells, and -1 for red and rose cells.
*   Termination: Episode ends when the agent reaches a terminal state (red or rose cell).

### 3. LunarLander

*   Observation Space: 8-dimensional vector representing the position, velocity, angle, and angular velocity of the lander, and whether each leg is in contact with the ground.
*   Action Space: Discrete actions (0: do nothing, 1: fire left engine, 2: fire main engine, 3: fire right engine).
*   Reward: +100 for landing safely, -100 for crashing, and small rewards for using fuel efficiently.
*   Termination: Episode ends when the lander crashes or comes to rest on the landing pad.

## Discussion and Further Exploration

*   **Comparison of Environments:** Compare the performance of DQN on the three different environments. How does the complexity of the environment affect the learning process?
*   **Impact of Parameters:** Experiment with different values of the learning rate, discount factor, exploration rate, and replay memory size to observe their effect on the learning process.
*   **Network Architecture:** Investigate different network architectures, such as adding more hidden layers or using convolutional layers for the GridWorld environment.
*   **Double DQN:** Implement the Double DQN algorithm to address the overestimation bias of the Q-learning algorithm.
*   **Dueling DQN:** Implement the Dueling DQN architecture to separate the value and advantage functions.
*   **Noisy DQN:** Implement the Noisy DQN algorithm to improve exploration.

## Notes

*   The code uses the PyTorch framework for implementing the DQN. Make sure you have PyTorch installed.
*   The GridWorld environment requires the `gridworld.py` file and the `gridworldPlans/` directory to be in the same directory as `dqn.py`.
*   This implementation provides a basic DQN solution for the three environments. You can further improve the performance by tuning the hyperparameters, using more advanced network architectures, or implementing other DQN extensions.
