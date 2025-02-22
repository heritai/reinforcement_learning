# -*- coding: utf-8 -*-
"""vae.py

This script implements a Variational Autoencoder (VAE) for the MNIST dataset, as part of a Reinforcement Learning course.
"""

from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

#-----------------------------------------------------------------------
# Configuration
#-----------------------------------------------------------------------
# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 200  # Batch size
latent_size = 20 # Latent vector size
hidden_size = 400  # Size of the hidden layer
learning_rate = 1e-3 # Learning rate
num_epochs = 100 # Number of epochs

# Output directory
output_dir = "results" # Directory to save the results

#-----------------------------------------------------------------------
# Data Loading
#-----------------------------------------------------------------------
# MNIST dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, pin_memory=True)

#-----------------------------------------------------------------------
# VAE Model Definition
#-----------------------------------------------------------------------
class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model.
    """
    def __init__(self, latent_size=20, hidden_size=400):
        """
        Initializes the VAE model.

        Args:
            latent_size (int): The dimension of the latent space.
            hidden_size (int): The size of the hidden layer.
        """
        super(VAE, self).__init__()

        self.latent_size = latent_size # Store latent vector size
        self.hidden_size = hidden_size # Store hidden layer size

        # Encoder layers
        self.fc1 = nn.Linear(784, hidden_size) # Input layer to hidden layer
        self.fc21 = nn.Linear(hidden_size, latent_size) # Hidden layer to mu
        self.fc22 = nn.Linear(hidden_size, latent_size) # Hidden layer to logvar

        # Decoder layers
        self.fc3 = nn.Linear(latent_size, hidden_size) # Latent vector to hidden layer
        self.fc4 = nn.Linear(hidden_size, 784)  # Hidden layer to output

    def encode(self, x):
        """
        Encodes the input into the latent space.

        Args:
            x (torch.Tensor): The input data (batch_size x 784).

        Returns:
            tuple: The mean (mu) and log variance (logvar) of the latent distribution.
        """
        h1 = F.relu(self.fc1(x)) # Apply ReLU activation
        return self.fc21(h1), self.fc22(h1) # Return mu and logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent distribution.

        Args:
            mu (torch.Tensor): The mean of the latent distribution.
            logvar (torch.Tensor): The log variance of the latent distribution.

        Returns:
            torch.Tensor: A sample from the latent distribution.
        """
        std = torch.exp(0.5*logvar) # Calculate the standard deviation
        eps = torch.randn_like(std) # Generate random noise
        return mu + eps*std # Sample from the distribution

    def decode(self, z):
        """
        Decodes the latent vector into the original data space.

        Args:
            z (torch.Tensor): The latent vector (batch_size x latent_size).

        Returns:
            torch.Tensor: The reconstructed data (batch_size x 784).
        """
        h3 = F.relu(self.fc3(z)) # Apply ReLU activation
        return torch.sigmoid(self.fc4(h3)) # Apply sigmoid activation

    def forward(self, x):
        """
        Forward pass through the VAE model.

        Args:
            x (torch.Tensor): The input data (batch_size x 1 x 28 x 28).

        Returns:
            tuple: The reconstructed data, the mean (mu), and the log variance (logvar).
        """
        mu, logvar = self.encode(x.view(-1, 784)) # Encode the input
        z = self.reparameterize(mu, logvar)  # Reparameterize to sample from the latent distribution
        return self.decode(z), mu, logvar # Decode the latent vector and return

#-----------------------------------------------------------------------
# Loss Function
#-----------------------------------------------------------------------
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    """
    Calculates the VAE loss function.

    Args:
        recon_x (torch.Tensor): The reconstructed data.
        x (torch.Tensor): The original data.
        mu (torch.Tensor): The mean of the latent distribution.
        logvar (torch.Tensor): The log variance of the latent distribution.

    Returns:
        torch.Tensor: The total loss.
    """
    # Binary cross-entropy reconstruction loss
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # KL divergence loss
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD # Return the total loss

#-----------------------------------------------------------------------
# Training and Testing Functions
#-----------------------------------------------------------------------
def train(epoch):
    """
    Trains the VAE model for one epoch.

    Args:
        epoch (int): The current epoch number.
    """
    model.train() # Set the model to training mode
    train_loss = 0 # Initialize the training loss
    for batch_idx, (data, _) in enumerate(train_loader):  # Iterate over the training data
        data = data.to(device)  # Move the data to the device
        optimizer.zero_grad() # Clear the gradients
        recon_batch, mu, logvar = model(data) # Pass the data through the model
        loss = loss_function(recon_batch, data, mu, logvar) # Calculate the loss
        loss.backward() # Calculate the gradients
        train_loss += loss.item() # Add the loss to the training loss
        optimizer.step() # Update the parameters

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data))) # Print the training progress

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset))) # Print the average training loss

def test(epoch):
    """
    Tests the VAE model for one epoch.

    Args:
        epoch (int): The current epoch number.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0 # Initialize the test loss
    with torch.no_grad(): # Disable gradient calculation
        for i, (data, _) in enumerate(test_loader): # Iterate over the test data
            data = data.to(device)  # Move the data to the device
            recon_batch, mu, logvar = model(data) # Pass the data through the model
            test_loss += loss_function(recon_batch, data, mu, logvar).item() # Calculate the loss
            if i == 0:
                n = min(data.size(0), 8) # Number of images to display
                comparison = torch.cat([data[:n],
                                      recon_batch.view(batch_size, 1, 28, 28)[:n]]) # Concatenate original and reconstructed images
                os.makedirs(output_dir, exist_ok=True)
                save_image(comparison.cpu(),
                         os.path.join(output_dir, 'reconstruction_' + str(epoch) + '.png'), nrow=n) # Save the images

    test_loss /= len(test_loader.dataset) # Calculate the average test loss
    print('====> Test set loss: {:.4f}'.format(test_loss)) # Print the test loss

#-----------------------------------------------------------------------
# Main Script
#-----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize the model and optimizer
    model = VAE(latent_size=latent_size, hidden_size=hidden_size).to(device)  # Create the VAE model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Create the Adam optimizer

    # Training loop
    for epoch in range(1, num_epochs + 1): # Iterate over the epochs

        train(epoch)  # Train the model
        test(epoch) # Test the model
        with torch.no_grad():  # Disable gradient calculation
            sample = torch.randn(64, latent_size).to(device)  # Generate random noise
            sample = model.decode(sample).cpu() # Decode the noise into an image
            os.makedirs(output_dir, exist_ok=True)
            save_image(sample.view(64, 1, 28, 28),
                       os.path.join(output_dir, 'sample_' + str(epoch) + '.png')) # Save the generated images

print("Training finished")
