# qlearning_dynaq: Q-learning, SARSA, and Dyna-Q in GridWorld

This project implements three reinforcement learning algorithms, Q-learning, SARSA, and Dyna-Q, for solving the GridWorld environment. It was created as part of a Reinforcement Learning course (TME3).

## Overview

The goal of this assignment is to understand and implement Q-learning, SARSA and Dyna-Q  to find the optimal policy for an agent navigating a grid-based world. The agent's objective is to reach a goal state while avoiding obstacles and negative reward states.

## Files

*   `qlearning_dynaq.py`:  The main Python script containing the implementations of the Q-learning, SARSA, and Dyna-Q algorithms, along with the GridWorld environment setup and training loops.
*   `gridworld.py`: (Assumed to be present) A custom environment for GridWorld.
*   `gridworldPlans/`: Contains various grid layouts (`planX.txt`) defining the environment.
*   `cumulative_reward_comparison.png`: Plot of cumulative rewards during training of the three algorithms.

## Environment Setup

1.  **Install Dependencies:**
    ```bash
    pip install gym matplotlib numpy
    ```
2.  **GridWorld Environment:**  Ensure that the `gridworld.py` file and the `gridworldPlans/` directory are in the same directory as `qlearning_dynaq.py`. These files define the GridWorld environment.

## Algorithms Implemented

### 1. Q-learning

*   An off-policy temporal difference learning algorithm.
*   Updates the Q-value based on the maximum possible reward in the next state, regardless of the action actually taken.

### 2. SARSA

*   An on-policy temporal difference learning algorithm.
*   Updates the Q-value based on the reward and the Q-value of the next state-action pair actually taken.

### 3. Dyna-Q

*   An extension of Q-learning that incorporates planning by using a model of the environment.
*   After each real experience, the agent updates its Q-value and also updates its model based on the transition. The agent then simulates `k` experiences by randomly selecting previously visited states and actions and updating the Q-values based on the model's predictions.

## Usage

1.  Run the script:
    ```bash
    python qlearning_dynaq.py
    ```

2.  The script will:
    *   Initialize the GridWorld environment.
    *   Train the Q-learning, SARSA, and Dyna-Q agents.
    *   Print the cumulative reward for each episode during training.
    *   Generate a plot of the cumulative rewards over episodes for all three algorithms, saved as `cumulative_reward_comparison.png`.

## Code Structure

*   **`BaseAgent` Class:** An abstract base class for reinforcement learning agents.
    *   `__init__(self, states, actions, lr=0.01, gamma=1, eps=0.3)`: Initializes the agent with the environment, discount factor (`gamma`), convergence threshold (`eps`), and exploration rate.
    *   `epsGreedpolicy(self, obs)`: Implements an epsilon-greedy policy for action selection.
    *   `act(self, obs)`: Selects an action based on the current state and the epsilon-greedy policy.
    *   `updateQvals(self, *args, **kwargs)`: Abstract method for updating Q-values.

*   **`QlearningAgent` Class:** Implements the Q-learning algorithm, inheriting from `BaseAgent`.
    *   `updateQvals(self, st0, st1, action, r, done)`: Updates the Q-value for the given state-action pair using the Q-learning update rule.

*   **`SARSAAgent` Class:** Implements the SARSA algorithm, inheriting from `BaseAgent`.
    *   `updateQvals(self, st0, st1, action1, action2, r, done)`: Updates the Q-value for the given state-action pair using the SARSA update rule.

*   **`DynaQAgent` Class:** Implements the Dyna-Q algorithm, inheriting from `BaseAgent`.
    *   `__init__(self, states, actions, terminal_states, lr=0.01, gamma=1, eps=0.3, k=10, alphaUpdate=0.01)`: Initializes the Dyna-Q agent with the model and planning parameters.
    *   `updateModel(self, st0, st1, action, r, alphaR=0.1)`: Updates the model based on the observed transition.
    *   `updatebyModel(self, alphaUpdate, k)`: Performs planning steps by updating the Q-values based on the model.
    *   `updateQvals(self, st0, st1, action, r, done)`: Updates the Q-value and the model.

## Key Parameters

*   `lr` (Learning Rate): Determines the step size for updating the Q-values.
*   `gamma` (Discount Factor): Determines the importance of future rewards.
*   `eps` (Exploration Rate): The probability of taking a random action.
*   `k` (Planning Steps): The number of planning steps performed by Dyna-Q after each real experience.
*   `alphaUpdate` (Model Learning Rate): The learning rate for updating the model in Dyna-Q.

## Discussion and Further Exploration

*   **Comparison of Algorithms:** Compare the performance of Q-learning, SARSA, and Dyna-Q in the GridWorld environment.  How do they differ in terms of convergence speed and final performance?
*   **Impact of Parameters:** Experiment with different values of the learning rate, discount factor, exploration rate, and planning steps to observe their effect on the learning process.
*   **GridWorld Plans:** Test the algorithms on different GridWorld plans (e.g., `plan1.txt`, `plan2.txt`, etc.) to evaluate their robustness.
*   **Eligibility Traces:** Implement eligibility traces to accelerate learning (as suggested in the bonus task of the assignment).

## Notes

*   The GridWorld environment converts the observation (which is a NumPy array) to a string to use it as a key in the Q-table.
*   The code includes plotting functionality to visualize the learning progress. Make sure you have `matplotlib` installed.
