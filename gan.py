# -*- coding: utf-8 -*-
"""gan.py

This script implements a Deep Convolutional Generative Adversarial Network (DCGAN) for face generation, as part of a Reinforcement Learning course.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

#-----------------------------------------------------------------------
# Configuration
#-----------------------------------------------------------------------
# Set random seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    cudnn.benchmark = True
    torch.cuda.manual_seed(seed)

# Data parameters
image_size = 64 # Image dimension
batch_size = 128 # Batch size
workers = 2 # Number of workers for data loading

# Model parameters
nz = 100 # Latent vector size
ngf = 64 # Generator feature map size
ndf = 64 # Discriminator feature map size
nc = 3  # Number of channels in the training images (RGB)

# Training parameters
num_epochs = 5  # Number of training epochs
lr = 0.0002 # Learning rate
beta1 = 0.5 # Beta1 hyperparameter for Adam optimizer

# Output directory
output_dir = "genFaces" # Directory to save generated images

#-----------------------------------------------------------------------
# Data Loading
#-----------------------------------------------------------------------
# Define the dataset and data loader
dataset = dset.ImageFolder(root="data/", # Root directory of the dataset
                           transform=transforms.Compose([  # Transformations to apply to the images
                               transforms.Resize(image_size), # Resize the images to image_size x image_size
                               transforms.CenterCrop(image_size),  # Crop the images to image_size x image_size
                               transforms.ToTensor(),  # Convert the images to PyTorch tensors
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalize the pixel values to the range [-1, 1]
                           ]))

# Create the data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Plot some training images
real_batch = next(iter(dataloader)) # Get a batch of real images

plt.figure(figsize=(15, 15))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=1, normalize=True).cpu(), (1, 2, 0))) # Create a grid of images
os.makedirs(output_dir, exist_ok=True) # Create the output directory if it doesn't exist
plt.savefig(os.path.join(output_dir, "train.png")) # Save the grid of images

#-----------------------------------------------------------------------
# Model Definition
#-----------------------------------------------------------------------
# Weight initialization function
def weights_init(m):
    """
    Initializes the weights of the neural network.

    Args:
        m (nn.Module): The neural network module.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02) # Initialize convolutional layer weights with a normal distribution
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02) # Initialize batch normalization layer weights with a normal distribution
        nn.init.constant_(m.bias.data, 0)  # Initialize batch normalization layer biases with a constant value

# Discriminator model
class Discriminator(nn.Module):
    """
    Discriminator network for distinguishing real and fake images.
    """
    def __init__(self):
        """
        Initializes the Discriminator network.
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # Convolutional layer
            nn.LeakyReLU(0.2, inplace=True), # Leaky ReLU activation
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), # Convolutional layer
            nn.BatchNorm2d(ndf * 2), # Batch normalization
            nn.LeakyReLU(0.2, inplace=True), # Leaky ReLU activation
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # Convolutional layer
            nn.BatchNorm2d(ndf * 4), # Batch normalization
            nn.LeakyReLU(0.2, inplace=True), # Leaky ReLU activation
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # Convolutional layer
            nn.BatchNorm2d(ndf * 8), # Batch normalization
            nn.LeakyReLU(0.2, inplace=True), # Leaky ReLU activation
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), # Convolutional layer
            nn.Sigmoid() # Sigmoid activation
        )

    def forward(self, input):
        """
        Forward pass through the network.

        Args:
            input (torch.Tensor): The input image (batch_size x nc x 64 x 64).

        Returns:
            torch.Tensor: The output probability (batch_size x 1 x 1 x 1) indicating whether the image is real or fake.
        """
        return self.main(input) # Apply the sequence of layers

# Generator model
class Generator(nn.Module):
    """
    Generator network for generating fake images from a latent vector.
    """
    def __init__(self):
        """
        Initializes the Generator network.
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False), # Transposed convolutional layer
            nn.BatchNorm2d(ngf * 8), # Batch normalization
            nn.ReLU(True),  # ReLU activation
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), # Transposed convolutional layer
            nn.BatchNorm2d(ngf * 4), # Batch normalization
            nn.ReLU(True), # ReLU activation
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False), # Transposed convolutional layer
            nn.BatchNorm2d(ngf * 2), # Batch normalization
            nn.ReLU(True), # ReLU activation
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False), # Transposed convolutional layer
            nn.BatchNorm2d(ngf), # Batch normalization
            nn.ReLU(True), # ReLU activation
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False), # Transposed convolutional layer
            nn.Tanh() # Tanh activation
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        """
        Forward pass through the network.

        Args:
            input (torch.Tensor): The input latent vector (batch_size x nz x 1 x 1).

        Returns:
            torch.Tensor: The generated image (batch_size x nc x 64 x 64).
        """
        return self.main(input)  # Apply the sequence of layers

# Alternative Generator model (as in the original DCGAN paper)
class GeneratorAlt(nn.Module):
    def __init__(self):
        super(GeneratorAlt, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Create the Discriminator
netD = Discriminator().to(device)
netD.apply(weights_init)
print(netD)

# Create the Generator
netG = Generator().to(device)
#netG = GeneratorAlt().to(device)  # Use the alternative generator
netG.apply(weights_init)
print(netG)

#-----------------------------------------------------------------------
# Training
#-----------------------------------------------------------------------
# Initialize loss function
criterion = nn.BCELoss()

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Training Loop

# Lists to keep track of progress
img_list = [] # List to store generated images
G_losses = [] # List to store generator losses
D_losses = [] # List to store discriminator losses
iters = 0 # Counter for iterations

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        #-----------------------------------------------------------------------
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        #-----------------------------------------------------------------------
        ## Train with all-real batch
        netD.zero_grad() # Clear the gradients
        # Format batch
        real_cpu = data[0].to(device) # Get real images from the batch
        b_size = real_cpu.size(0) # Get the batch size
        label = torch.full((b_size,), real_label, device=device).float() # Create a tensor of real labels

        # Forward pass real batch through D
        output = netD(real_cpu).view(-1) # Pass real images through the discriminator
        # Calculate loss on all-real batch
        errD_real = criterion(output, label) # Calculate the loss for the real batch

        # Calculate gradients for D in backward pass
        errD_real.backward() # Calculate the gradients
        D_x = output.mean().item() # Get the mean of the discriminator output for real images

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device) # Generate a batch of random noise

        # Generate fake image batch with G
        fake = netG(noise) # Generate fake images using the generator
        label.fill_(fake_label) # Create the labels for the fake images

        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1) # Pass the fake images through the discriminator
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label) # Calculate the loss for the fake batch
        # Calculate the gradients for this batch
        errD_fake.backward() # Calculate the gradients
        D_G_z1 = output.mean().item() # Get the mean of the discriminator output for fake images

        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake # Calculate the total discriminator loss
        # Update D
        optimizerD.step() # Update the discriminator parameters

        #-----------------------------------------------------------------------
        # (2) Update G network: maximize log(D(G(z)))
        #-----------------------------------------------------------------------
        netG.zero_grad() # Clear the gradients
        label.fill_(real_label)  # Fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1) # Pass the fake images through the discriminator

        # Calculate G's loss based on this output
        errG = criterion(output, label) # Calculate the generator loss
        # Calculate gradients for G
        errG.backward() # Calculate the gradients
        D_G_z2 = output.mean().item() # Get the mean of the discriminator output for fake images after generator update

        # Update G
        optimizerG.step() # Update the generator parameters

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()  # Generate fake images using the fixed noise
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))  # Create a grid of fake images

        iters += 1 # Increment the iteration counter

#-----------------------------------------------------------------------
# Results Visualization
#-----------------------------------------------------------------------
# Generate fake images after training
noise = torch.randn(64, nz, 1, 1, device=device) # Generate a batch of random noise
with torch.no_grad():
    netG.eval() # Set the generator to evaluation mode
    fake = netG(noise).detach().cpu() # Generate fake images
img = vutils.make_grid(fake, padding=2, normalize=True) # Create a grid of fake images
img_list.append(img) # Append the grid of fake images to the list

# Plot fake images
plt.figure(figsize=(15, 15))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img.cpu(), (1, 2, 0))) # Display the grid of fake images
plt.savefig(os.path.join(output_dir, "fake.png")) # Save the grid of fake images

# Plot generator and discriminator loss during training
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G") # Plot the generator loss
plt.plot(D_losses, label="D") # Plot the discriminator loss
plt.xlabel("iterations") # Set the x-axis label
plt.ylabel("Loss") # Set the y-axis label
plt.legend() # Show the legend
plt.savefig(os.path.join(output_dir, "loss.png"))
plt.show() # Show the plot
