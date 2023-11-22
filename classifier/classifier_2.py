import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
import os

# ************* Create custom Dataset ************* #
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = self.data.iloc[index]
        y = self.labels.iloc[index]
        return x, y

# Create training and test data
train_data = pd.read_parquet("/work/tfluehma/git/SSNF2/classifier/data/train_data.parquet")
test_data = pd.read_parquet("/work/tfluehma/git/SSNF2/classifier/data/test_data.parquet")

train = train_data.iloc[:, :-1]
train_label = train_data["label"]

test = test_data.iloc[:, :-1]
test_label = test_data["label"]

train_dataset = CustomDataset(train, train_label)
test_dataset = CustomDataset(test, test_label)

# Create Dataloader
batch_size=64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# ************** Build the Neural Network ******************
# Get device for training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


len = len(train_dataloader)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(len, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x = self.fatten(x)
        x = self.linear_relu_stack(x)
        return x
    
model = NeuralNetwork().to(device)
print(model)

# Define Hyperparameters
learning_rate = 1e-3
epochs = 5
batch_size = 64



# **************** Train Function **************** #
def train_loop(dataloader, model, loss_fn, optimizer):
    print("here")
    print(len(dataloader.dataset))
    size = len(dataloader.dataset)
    print("hey")
    # Set the model to training mode
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# **************** Test Function **************** #
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluation the model with torch.no_grad() ensures that no gradients
    # are computed during test mode
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item
            correct += (pred.argmax(1)== y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    
# initialize the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10

# ********************* Actual Testing ********************* #
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

print("Done")
