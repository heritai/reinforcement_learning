# policy_value_iteration.py: Policy Iteration and Value Iteration in GridWorld

This project implements and compares two dynamic programming algorithms, Policy Iteration and Value Iteration, for solving the GridWorld environment.  It was created as part of a Reinforcement Learning course (TME2).

## Overview

The goal of this assignment is to understand and implement Policy Iteration and Value Iteration to find the optimal policy for an agent navigating a grid-based world. The agent's objective is to reach a goal state while avoiding obstacles and negative reward states.

## Files

*   `policy_value_iteration.py`:  The main Python script containing the implementations of the Policy Iteration and Value Iteration algorithms, along with the GridWorld environment setup and training loops.
*   `gridworld.py`: (Assumed to be present) A custom environment for GridWorld.
*   `gridworldPlans/`: Contains various grid layouts (`planX.txt`) defining the environment.
*   `policy_iteration_cumulative_reward.png`: Plot of cumulative rewards during policy iteration.
*   `value_iteration_cumulative_reward.png`: Plot of cumulative rewards during value iteration.

## Environment Setup

1.  **Install Dependencies:**
    ```bash
    pip install gym matplotlib numpy
    ```
2.  **GridWorld Environment:**  Ensure that the `gridworld.py` file and the `gridworldPlans/` directory are in the same directory as `policy_value_iteration.py`. These files define the GridWorld environment.
    ```

## Algorithms Implemented

### 1. Policy Iteration

*   **Policy Evaluation:** Iteratively computes the value function for a given policy until convergence.
*   **Policy Improvement:**  Greedily updates the policy by selecting the action that maximizes the expected return for each state, based on the evaluated value function.
*   The algorithm alternates between policy evaluation and policy improvement until the policy converges to the optimal policy.

### 2. Value Iteration

*   Iteratively updates the value function for each state by considering the maximum expected return achievable from that state, assuming optimal actions are taken in the future.
*   After the value function converges, the optimal policy is extracted by selecting the action that maximizes the expected return for each state.

## Usage

1.  Run the script:
    ```bash
    python policy_value_iteration.py
    ```

2.  The script will:
    *   Initialize the GridWorld environment.
    *   Train both the Policy Iteration and Value Iteration agents.
    *   Print the cumulative reward for each episode during training.
    *   Generate plots of the cumulative rewards over episodes for both algorithms, saved as  `policy_iteration_cumulative_reward.png` and `value_iteration_cumulative_reward.png`.

## Code Structure

*   **`PolicyIterAgent` Class:** Implements the Policy Iteration algorithm.
    *   `__init__(self, env, gamma=1, eps=0.0001, exploration_rate=0.1)`: Initializes the agent with the environment, discount factor (`gamma`), convergence threshold (`eps`), and exploration rate.
    *   `policyEval(self, pol)`:  Evaluates the given policy `pol` and returns the value function.
    *   `nextPolicy(self)`: Improves the current policy based on the value function obtained from `policyEval`.  Uses an epsilon-greedy approach for exploration.
    *   `act(self, observation)`: Returns the action to take based on the current policy.

*   **`ValueIterAgent` Class:** Implements the Value Iteration algorithm.
    *   `__init__(self, env, gamma=1, eps=0.0001)`: Initializes the agent with the environment, discount factor (`gamma`), and convergence threshold (`eps`).
    *   `valueEval(self)`: Performs one iteration of Value Iteration, updating the value function.
    *   `nextPolicy(self)`: Extracts the optimal policy from the converged value function.
    *   `act(self, observation)`: Returns the action to take based on the current policy.

## Key Parameters

*   `gamma` (Discount Factor): Determines the importance of future rewards. A higher value of `gamma` gives more weight to future rewards.
*   `eps` (Convergence Threshold):  The threshold for determining when the value function or policy has converged.  A smaller value results in more iterations but potentially a more accurate solution.
*   `exploration_rate`: The probability of taking a random action in Policy Iteration.  Helps to avoid local optima.

## Discussion and Further Exploration

*   **Comparison of Algorithms:** Compare the convergence speed and performance of Policy Iteration and Value Iteration. How do they perform with different grid layouts and reward structures?
*   **Impact of Gamma:** Experiment with different values of `gamma` to observe its effect on the learned policy and value function.
*   **GridWorld Plans:** Test the algorithms on different GridWorld plans (e.g., `plan1.txt`, `plan2.txt`, etc.) to evaluate their robustness.
*   **Exploration Rate:** Tuning the exploration rate can impact how well policy iteration converges.

## Notes

*   The GridWorld environment converts the observation (which is a NumPy array) to a string to use it as a key in the policy and value function dictionaries.
*   The code includes plotting functionality to visualize the learning progress. Make sure you have `matplotlib` installed.
