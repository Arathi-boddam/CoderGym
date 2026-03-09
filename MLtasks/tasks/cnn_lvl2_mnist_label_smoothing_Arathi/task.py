import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_task_metadata():
    return {
        "task_name": "cifar10_dropout_cnn",
        "task_type": "classification",
        "num_classes": 10,
        "input_shape": [3,32,32]
    }

class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            nn.Flatten(),
            nn.Linear(64*8*8,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,10)
        )

    def forward(self,x):

        x = self.conv(x)
        x = self.fc(x)

        return x

def make_dataloaders():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    train = datasets.CIFAR10("./data",train=True,download=True,transform=transform)
    test = datasets.CIFAR10("./data",train=False,download=True,transform=transform)

    train_loader = DataLoader(train,batch_size=64,shuffle=True)
    test_loader = DataLoader(test,batch_size=64)

    return train_loader,test_loader

def main():

    train_loader,test_loader = make_dataloaders()

    model = CNN().to(device)

    opt = optim.Adam(model.parameters(),lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):

        model.train()

        for x,y in train_loader:

            x,y = x.to(device),y.to(device)

            opt.zero_grad()

            out = model(x)

            loss = loss_fn(out,y)

            loss.backward()

            opt.step()

    correct=0
    total=0

    model.eval()

    with torch.no_grad():

        for x,y in test_loader:

            x,y = x.to(device),y.to(device)

            pred = torch.argmax(model(x),1)

            correct += (pred==y).sum().item()

            total += y.size(0)

    acc = correct/total

    print("Accuracy:",acc)

    if acc > 0.6:
        return 0
    return 1

if __name__=="__main__":
    sys.exit(main())
    