# gan: DCGAN Implementation for Face Generation

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) for generating face images from the CelebA dataset, as part of a Reinforcement Learning course (TME9).

## Overview

The goal of this assignment is to understand and implement GANs for generating realistic images. This project focuses on the DCGAN architecture, which uses convolutional neural networks for both the generator and discriminator.

## Files

*   `gan.py`: The main Python script containing the DCGAN implementation, including the generator and discriminator networks, training loop, and data loading.
*   `data/`: This directory is expected to contain the CelebA dataset (images). The directory structure should follow the format expected by `torchvision.datasets.ImageFolder`.  Specifically, put the `img_align_celeba` directory that comes with the CelebA dataset inside this `data` directory.
*   `genFaces/`: This directory will be created by the script to store the generated images and loss plots.

## Environment Setup

1.  **Install Dependencies:**
    ```bash
    pip install gym matplotlib numpy torch torchvision
    ```
2.  **Download and Prepare the CelebA Dataset:**
    *   Download the CelebA dataset from a reliable source (e.g., the link provided in the assignment description).  You'll likely need to create an account to download it.
    *   Extract the `img_align_celeba.zip` file.  This will create a directory called `img_align_celeba` containing the images.
    *   Create a directory called `data` in the same directory as `gan.py`.
    *   Move the `img_align_celeba` directory into the `data` directory.  The final directory structure should look like this:
        ```
        gan.py
        data/
            img_align_celeba/
                000001.jpg
                000002.jpg
                ...
        ```

## Algorithms Implemented

### Deep Convolutional Generative Adversarial Network (DCGAN)

*   A type of GAN that uses convolutional neural networks for both the generator and discriminator.
*   The generator network learns to generate realistic images from a latent vector.
*   The discriminator network learns to distinguish between real and generated images.
*   The generator and discriminator are trained adversarially, with the generator trying to fool the discriminator and the discriminator trying to correctly classify real and generated images.

## Usage

1.  Run the script:
    ```bash
    python gan.py
    ```

2.  The script will:
    *   Load the CelebA dataset.
    *   Define the generator and discriminator networks.
    *   Train the DCGAN model.
    *   Generate fake images and save them to the `genFaces/` directory.
    *   Plot the generator and discriminator loss during training and save the plot to the `genFaces/` directory.

## Code Structure

*   **`weights_init` Function:** Initializes the weights of the neural network modules.
*   **`Discriminator` Class:** Implements the discriminator network.
    *   `__init__(self)`: Initializes the discriminator network layers.
    *   `forward(self, input)`: Performs a forward pass through the network to compute the probability that the input image is real.
*   **`Generator` Class:** Implements the generator network.
    *   `__init__(self)`: Initializes the generator network layers.
    *   `forward(self, input)`: Performs a forward pass through the network to generate an image from the input latent vector.
*   **Training Loop:** Implements the training loop for the DCGAN model.
    *   Loads a batch of real images from the dataset.
    *   Generates a batch of fake images using the generator.
    *   Trains the discriminator to distinguish between real and fake images.
    *   Trains the generator to generate images that can fool the discriminator.
    *   Logs the losses for the generator and discriminator.
    *   Saves generated images at the end of each epoch to visualize the training progress.

## Key Parameters

*   `image_size`: The size of the input images (64x64).
*   `batch_size`: The number of images in each batch.
*   `nz`: The size of the latent vector (100).
*   `ngf`: The number of feature maps in the generator.
*   `ndf`: The number of feature maps in the discriminator.
*   `num_epochs`: The number of training epochs.
*   `lr`: The learning rate for the Adam optimizers.
*   `beta1`: The beta1 hyperparameter for the Adam optimizers.

## Discussion and Further Exploration

*   **Experiment with Hyperparameters:** Tune the hyperparameters (e.g., learning rate, batch size, number of epochs) to improve the quality of the generated images.
*   **Try Different Architectures:** Experiment with different network architectures for the generator and discriminator, such as adding more layers or using different activation functions.
*   **Implement Conditional GANs:** Implement a conditional GAN (CGAN) to generate images with specific attributes (e.g., generate faces with specific hair color or gender).
*   **Evaluate the Generated Images:** Use quantitative metrics (e.g., Inception Score, Fréchet Inception Distance (FID)) to evaluate the quality of the generated images.  These are more complex to implement, but provide a more objective measure than visual inspection.

## Notes

*   The code uses the PyTorch framework for implementing the GAN. Make sure you have PyTorch installed.
*   The CelebA dataset is large, so make sure you have enough disk space to store it.
*   Training GANs can be challenging and may require significant computational resources.
*   The quality of the generated images may vary depending on the dataset and the network architecture.
