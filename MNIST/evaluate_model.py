import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score
import struct
import numpy as np
from train_model import ImageClassifier 


def evaluate_model(model, test_loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output, 1)


            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')

        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")


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
    
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Paths
testing_images_filepath = r"dataset\t10k-images-idx3-ubyte\t10k-images-idx3-ubyte"
testing_labels_filepath = r"dataset\t10k-labels-idx1-ubyte\t10k-labels-idx1-ubyte"

# Load data
test_images = load_images(testing_images_filepath)
test_labels = load_labels(testing_labels_filepath)

# Convert to tensors
test_images_tensor = torch.tensor(test_images, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

# Wrap into dataset
test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load model
model = ImageClassifier().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))

# Evaluate
evaluate_model(model, test_loader, device)