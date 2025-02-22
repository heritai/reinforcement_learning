# vae: Variational Autoencoder for MNIST

This project implements a Variational Autoencoder (VAE) for generating MNIST handwritten digit images, as part of a Reinforcement Learning course (TME10).

## Overview

The goal of this assignment is to understand and implement VAEs as generative models. VAEs learn a latent space representation of the data and then generate new data points by sampling from this latent space.

## Files

*   `vae.py`: The main Python script containing the VAE implementation, including the encoder and decoder networks, the loss function, the training loop, and data loading.
*   `results/`: This directory will be created by the script to store the generated and reconstructed images.

## Environment Setup

1.  **Install Dependencies:**
    ```bash
    pip install gym matplotlib numpy torch torchvision
    ```

## Algorithms Implemented

### Variational Autoencoder (VAE)

*   A generative model that learns a probabilistic latent space representation of the data.
*   Consists of an encoder network that maps the input data to a latent distribution (typically a Gaussian distribution) and a decoder network that maps samples from the latent distribution back to the original data space.
*   The model is trained to minimize a loss function that consists of two terms: a reconstruction loss that measures how well the decoder can reconstruct the input data from the latent representation, and a Kullback-Leibler (KL) divergence loss that measures how close the learned latent distribution is to a prior distribution (typically a standard Gaussian).

## Usage

1.  Run the script:
    ```bash
    python vae.py
    ```

2.  The script will:
    *   Download the MNIST dataset.
    *   Define the VAE model (encoder and decoder networks).
    *   Train the VAE model.
    *   Generate sample images from the latent space and save them to the `results/` directory.
    *   Reconstruct images from the test set and save the original and reconstructed images to the `results/` directory.

## Code Structure

*   **`VAE` Class:** Implements the VAE model.
    *   `__init__(self, latent_size=20, hidden_size=400)`: Initializes the VAE model with a specified latent size and hidden size.
    *   `encode(self, x)`: Encodes the input data into the latent space, returning the mean (mu) and log variance (logvar) of the latent distribution.
    *   `reparameterize(self, mu, logvar)`: Implements the reparameterization trick to sample from the latent distribution.
    *   `decode(self, z)`: Decodes the latent vector into the original data space.
    *   `forward(self, x)`: Performs a forward pass through the VAE model.
*   **`loss_function` Function:** Calculates the VAE loss function, including the reconstruction loss (binary cross-entropy) and the KL divergence loss.
*   **`train` Function:** Trains the VAE model for one epoch.
*   **`test` Function:** Tests the VAE model and saves reconstructed images.

## Key Parameters

*   `batch_size`: The number of images in each batch.
*   `latent_size`: The dimension of the latent vector.
*   `hidden_size`: The size of the hidden layer in the encoder and decoder networks.
*   `learning_rate`: The learning rate for the Adam optimizer.
*   `num_epochs`: The number of training epochs.

## Discussion and Further Exploration

*   **Experiment with Hyperparameters:** Tune the hyperparameters (e.g., learning rate, batch size, latent size, hidden size) to improve the quality of the generated images.
*   **Try Different Architectures:** Experiment with different network architectures for the encoder and decoder, such as adding more layers, using convolutional layers, or using different activation functions.
*   **Visualize the Latent Space:** Visualize the latent space by plotting the embeddings of different digits. Use dimensionality reduction techniques (e.g., PCA, t-SNE) to reduce the latent space to 2 or 3 dimensions for visualization.
*   **Conditional VAEs:** Implement a conditional VAE (CVAE) to generate images with specific labels.

## Notes

*   The code uses the PyTorch framework for implementing the VAE. Make sure you have PyTorch installed.
*   The MNIST dataset will be downloaded automatically by the script.
*   The generated and reconstructed images will be saved to the `results/` directory.
