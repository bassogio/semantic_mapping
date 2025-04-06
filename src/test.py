import torch
import torch.nn as nn

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Create a random input tensor of shape (batch_size, channels, height, width)
input_tensor = torch.randn(1, 3, 32, 32, device=device)

# Define a convolutional layer.
# This layer will use cuDNN for its operations if available.
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1).to(device)

# Perform the forward pass
output_tensor = conv_layer(input_tensor)

# Print the output shape
print("Output shape:", output_tensor.shape)
