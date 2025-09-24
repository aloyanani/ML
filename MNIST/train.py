import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn

# Define transformations to apply to the images
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.1307,), (0.3081,)) # Normalize the pixel values
])

# Load the training dataset
train_dataset = torchvision.datasets.MNIST(root='./MNIST/data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Load the test dataset
test_dataset = torchvision.datasets.MNIST(root='./MNIST/data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Get one batch of training data
train_data, labels = next(iter(train_loader))
print("Images batch shape:", train_data.shape)  # [batch_size, channels, height, width]
print("Labels batch shape:", labels.shape)  # [batch_size]

# Define a simple CNN model

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    nn.Flatten(),
    nn.Linear(16, 10)       
)