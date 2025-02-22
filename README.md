# reinforcement_learning

This repository contains a collection of reinforcement learning algorithms implemented in Python using PyTorch. It was created to fulfill the requirements of a Reinforcement Learning course. The algorithms cover a range of topics, from dynamic programming to deep reinforcement learning, and are tested on various environments and datasets.

## Overview

This repository is organized into several Python scripts, each implementing a specific reinforcement learning algorithm. Each script also has a corresponding markdown file in the `/readme` directory that provides a detailed explanation of the code, its usage, and the underlying algorithm.

## Files

The repository contains the following files:

| File                       | Description                                                                                                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ddpg.py`                  | Implements the Deep Deterministic Policy Gradient (DDPG) algorithm for continuous control environments.                                                              |
| `dqn.py`                   | Implements the Deep Q-Network (DQN) algorithm for discrete action spaces.                                                                                               |
| `gan.py`                   | Implements a Deep Convolutional Generative Adversarial Network (DCGAN) for generating face images from the CelebA dataset.                                            |
| `multiagent_ddpg.py`       | Implements the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm for cooperative multi-agent environments.                                           |
| `policy_gradient.py`        | Implements the Advantage Actor-Critic (A2C) algorithm for policy gradients.                                                                                             |
| `policy_value_iteration.py` | Implements Policy Iteration and Value Iteration algorithms for solving Markov Decision Processes (MDPs), specifically the GridWorld environment.                          |
| `qlearning_dynaq.py`        | Implements Q-learning, SARSA, and Dyna-Q algorithms for solving the GridWorld environment.                                                                               |
| `vae.py`                   | Implements a Variational Autoencoder (VAE) for generating MNIST handwritten digit images.                                                                              |
| `vae.md`                  | Detailed explanation of the VAE code.                                                                      |
| `ddpg.md` | Detailed explanation of the DDPG code.  |
| `dqn.md`      |Detailed explanation of the DQN code.     |
|`gan.md`| Detailed explanation of the GAN code.     |
|`multiagent_ddpg.md`| Detailed explanation of the MADDPG code.   |
|`policy_gradient.md`| Detailed explanation of the A2C (Policy Gradient) code.  |
|`policy_value_iteration.md`| Detailed explanation of the Policy and Value Iteration code. |
|`qlearning_dynaq.md`| Detailed explanation of the Q-Learning, SARSA, and Dyna-Q algorithms.|

## Algorithms and Environments

Here's a brief overview of the algorithms and environments used in the repository:

*   **Dynamic Programming:**
    *   **Policy Iteration:** An algorithm for finding the optimal policy in an MDP by iteratively evaluating and improving the policy.
    *   **Value Iteration:** An algorithm for finding the optimal value function and policy in an MDP by iteratively updating the value function.
    *   **Environment:** GridWorld
*   **Tabular Reinforcement Learning:**
    *   **Q-learning:** An off-policy temporal difference learning algorithm that learns the optimal Q-function.
    *   **SARSA:** An on-policy temporal difference learning algorithm that learns the Q-function for the current policy.
    *   **Dyna-Q:** An extension of Q-learning that incorporates planning by using a model of the environment.
    *   **Environment:** GridWorld
*   **Policy Gradient Methods:**
    *   **Advantage Actor-Critic (A2C):** A policy gradient method that uses an actor to learn the optimal policy and a critic to estimate the value function.
    *   **Environment:** CartPole
*   **Deep Reinforcement Learning:**
    *   **Deep Q-Network (DQN):** A Q-learning algorithm that uses a deep neural network to approximate the Q-function.
    *   **Environments:** CartPole, GridWorld, LunarLander
    *   **Deep Deterministic Policy Gradient (DDPG):** An actor-critic algorithm for continuous control environments that uses deep neural networks to approximate the policy and Q-function.
    *   **Environments:** MountainCarContinuous, LunarLanderContinuous, Pendulum
    *   **Multi-Agent Deep Deterministic Policy Gradient (MADDPG):** An extension of DDPG to handle multi-agent environments with cooperative or competitive agents.
    *   **Environments:** Multi-Agent Particle Environments (e.g., simple\_spread, simple\_adversary, simple\_tag)
*   **Generative Models:**
    *   **Generative Adversarial Network (GAN):** A framework for training generative models by pitting two neural networks against each other: a generator that tries to create realistic data and a discriminator that tries to distinguish between real and generated data.
    *   **Dataset:** CelebA (face images)
    *   **Variational Autoencoder (VAE):** A probabilistic generative model that learns a latent space representation of the data and then generates new data points by sampling from this latent space.
    *   **Dataset:** MNIST (handwritten digits)

## Getting Started

1.  Clone the repository:
    ```bash
    git clone [repository URL]
    cd reinforcement_learning
    ```

2.  Install the required dependencies:
    ```bash
    pip install gym matplotlib numpy torch torchvision
    ```

3.  Explore the code and documentation for each algorithm and environment.

## Usage

To run the code for a specific algorithm and environment, navigate to the corresponding Python script and execute it:

```bash
python dqn.py  # Example: Run the DQN algorithm on the CartPole environment
```

Refer to the comments in the code and the `/readme` markdown files for specific instructions on how to run and configure each algorithm.

## Contributing

Contributions to this repository are welcome! If you find a bug, have a suggestion for improvement, or want to add a new algorithm or environment, please submit a pull request.

## License

This project is licensed under the [License Name] License.

