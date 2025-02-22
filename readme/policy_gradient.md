# policy_gradient: A2C Implementation for Policy Gradients

This project implements the Advantage Actor-Critic (A2C) algorithm for policy gradients, as part of a Reinforcement Learning course (TME5).

## Overview

The goal of this assignment is to implement and evaluate the A2C algorithm, a policy gradient method that combines an actor (policy) and a critic (value function) to improve learning stability and performance.

## Files

*   `policy_gradient.py`: The main Python script containing the A2C implementation, including the network architecture, training loop, and environment setup.

## Environment Setup

1.  **Install Dependencies:**
    ```bash
    pip install gym matplotlib numpy torch pandas
    ```

## Algorithms Implemented

### Advantage Actor-Critic (A2C)

*   A policy gradient method that uses an actor to learn the optimal policy and a critic to estimate the value function.
*   The actor updates the policy based on the advantage function, which measures the relative benefit of taking a particular action in a given state compared to the average action.
*   The critic learns to predict the value function, which estimates the expected cumulative reward for a given state.
*   A2C is a synchronous, on-policy algorithm where multiple agents collect experiences in parallel and update a central policy.

## Usage

1.  Run the script:
    ```bash
    python policy_gradient.py
    ```

2.  The script will:
    *   Create the CartPole environment.
    *   Train the A2C agent.
    *   Generate plots of the episode rewards and episode lengths over time, displaying the learning progress.

## Code Structure

*   **`ActorCritic` Class:** Implements the Actor-Critic network architecture.
    *   `__init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4)`: Initializes the ActorCritic network with a specified input size, output size, hidden size, and learning rate.
    *   `forward(self, state)`: Performs a forward pass through the network to compute the value estimate and policy distribution for a given state.

*   **`a2c` Function:** Implements the A2C training algorithm.
    *   Takes the environment as input.
    *   Initializes the ActorCritic network, optimizer, and other training parameters.
    *   Implements the training loop, including action selection, environment interaction, advantage calculation, and network updates.
    *   Generates plots of the episode rewards and episode lengths to visualize the learning progress.

## Key Parameters

*   `HIDDEN_SIZE`: The number of units in the hidden layers of the ActorCritic network.
*   `LEARNING_RATE`: The learning rate for the optimizer.
*   `GAMMA`: The discount factor, which determines the importance of future rewards.
*   `NUM_STEPS`: The maximum number of steps per episode.
*   `MAX_EPISODES`: The maximum number of episodes to train for.

## Discussion and Further Exploration

*   **Environments:** Test the A2C implementation on different environments, such as the LunarLander environment or other OpenAI Gym environments.
*   **Hyperparameter Tuning:** Experiment with different values of the hyperparameters, such as the learning rate, hidden size, and discount factor, to observe their effect on the learning process.
*   **Network Architecture:** Investigate different network architectures for the ActorCritic network, such as adding more hidden layers, using different activation functions, or incorporating convolutional layers.
*   **Advantage Function:** Explore different methods for estimating the advantage function, such as using Generalized Advantage Estimation (GAE).
*   **Entropy Regularization:** Tune the entropy regularization term to encourage exploration and improve the stability of the learning process.

## Notes

*   The code uses the PyTorch framework for implementing the ActorCritic network. Make sure you have PyTorch installed.
*   This implementation provides a basic A2C solution for the CartPole environment. You can further improve the performance by tuning the hyperparameters, using more advanced network architectures, or implementing other A2C extensions.
