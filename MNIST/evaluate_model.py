import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from sklearn.metrics import f1_score, precision_score
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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Load the test dataset# Define transformations to apply to the images
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.1307,), (0.3081,)) # Normalize the pixel values
])

   
# Load the test dataset
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

model = ImageClassifier()                 # create model instance
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))  # load weights
model.to(device)
model.eval()                              # set model to evaluation mode

evaluate_model(model, test_loader, device)