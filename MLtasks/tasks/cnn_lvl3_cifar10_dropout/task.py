import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_task_metadata():
    return {
        "task_name": "fashion_mnist_cnn",
        "task_type": "classification",
        "num_classes": 10,
        "input_shape": [1,28,28],
        "description": "CNN classifier on FashionMNIST"
    }

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        return self.net(x)

def make_dataloaders(batch_size=64):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train = datasets.FashionMNIST("./data",train=True,download=True,transform=transform)
    test = datasets.FashionMNIST("./data",train=False,download=True,transform=transform)

    train_loader = DataLoader(train,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test,batch_size=batch_size)

    return train_loader,test_loader

def build_model():
    return CNN().to(device)

def train(model,loader,epochs=5):

    optimizer = optim.Adam(model.parameters(),lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):

        for x,y in loader:

            x,y = x.to(device),y.to(device)

            optimizer.zero_grad()

            out = model(x)

            loss = loss_fn(out,y)

            loss.backward()

            optimizer.step()

def evaluate(model,loader):

    model.eval()

    correct=0
    total=0

    with torch.no_grad():

        for x,y in loader:

            x,y = x.to(device),y.to(device)

            out = model(x)

            pred = torch.argmax(out,1)

            correct += (pred==y).sum().item()

            total += y.size(0)

    return correct/total

def main():

    train_loader,test_loader = make_dataloaders()

    model = build_model()

    train(model,train_loader)

    acc = evaluate(model,test_loader)

    print("Test Accuracy:",acc)

    if acc > 0.85:
        return 0
    return 1

if __name__=="__main__":
    sys.exit(main())