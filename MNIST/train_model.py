import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score
import struct
import numpy as np
from torch.utils.data import TensorDataset, DataLoader



# Define a CNN model
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    nn.Flatten(),
                    nn.Linear(16*3*3, 10)       
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return x
    



def train_model(model, train_loader, optimizer, loss_f, device, epochs=10):

    for epoch in range(epochs):  
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # Reset gradients
            outputs = model(images)  # Forward pass
            loss = loss_f(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

    
        print(f"Epoch:{epoch} loss is {loss.item()}")

    # Save model after training
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Model saved as mnist_cnn.pth..")


def load_images(filepath):
    with open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
        return data.astype(np.float32) / 255.0  # normalize

def load_labels(filepath):
    with open(filepath, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    

    # # Load the test dataset# Define transformations to apply to the images
    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # Convert images to PyTorch tensors
    #     transforms.Normalize((0.1307,), (0.3081,)) # Normalize the pixel values
    # ])

    # # Load the training dataset
    # train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

   # Paths
    training_images_filepath = r"dataset/train-images-idx3-ubyte\train-images-idx3-ubyte"
    training_labels_filepath = r"dataset/train-labels-idx1-ubyte\train-labels-idx1-ubyte"

    # Load numpy arrays
    train_images = load_images(training_images_filepath)
    train_labels = load_labels(training_labels_filepath)

    # Convert to tensors
    train_images_tensor = torch.tensor(train_images, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)

    # Use TensorDataset
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Model
    model = ImageClassifier().to(device)
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    train_model(model, train_loader, optimizer, loss_f, device, epochs=10)
    

if __name__ == "__main__":
    main()