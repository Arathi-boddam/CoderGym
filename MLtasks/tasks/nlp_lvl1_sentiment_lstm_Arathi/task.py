import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.net=nn.Sequential(

            nn.Flatten(),

            nn.Linear(784,256),
            nn.ReLU(),

            nn.Linear(256,10)
        )

    def forward(self,x):

        return self.net(x)

def main():

    transform=transforms.ToTensor()

    train=datasets.MNIST("./data",train=True,download=True,transform=transform)
    test=datasets.MNIST("./data",train=False,download=True,transform=transform)

    train_loader=DataLoader(train,batch_size=64,shuffle=True)
    test_loader=DataLoader(test,batch_size=64)

    model=MLP().to(device)

    opt=optim.Adam(model.parameters(),lr=0.001)

    loss_fn=nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(5):

        for x,y in train_loader:

            x,y=x.to(device),y.to(device)

            opt.zero_grad()

            loss=loss_fn(model(x),y)

            loss.backward()

            opt.step()

    correct=0
    total=0

    with torch.no_grad():

        for x,y in test_loader:

            x,y=x.to(device),y.to(device)

            pred=torch.argmax(model(x),1)

            correct+=(pred==y).sum().item()

            total+=y.size(0)

    acc=correct/total

    print("Accuracy:",acc)

    if acc>0.97:
        return 0
    return 1

if __name__=="__main__":
    sys.exit(main())