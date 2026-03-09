import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_task_metadata():
    return {
        "task_name": "california_housing_regression",
        "task_type": "regression",
        "input_features": 8,
        "description": "Predict median house value using MLP"
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return device


def make_dataloaders(batch_size=64):

    data = fetch_california_housing()

    X = data.data
    y = data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

    train_loader = DataLoader(
        TensorDataset(X_train,y_train),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test,y_test),
        batch_size=batch_size
    )

    return train_loader, test_loader


def build_model():

    model = nn.Sequential(

        nn.Linear(8,64),
        nn.ReLU(),

        nn.Linear(64,32),
        nn.ReLU(),

        nn.Linear(32,1)

    )

    return model.to(device)


def train(model,loader,epochs=100):

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    for epoch in range(epochs):

        model.train()

        for X,y in loader:

            X,y = X.to(device),y.to(device)

            optimizer.zero_grad()

            pred = model(X)

            loss = criterion(pred,y)

            loss.backward()

            optimizer.step()


def evaluate(model,loader):

    model.eval()

    preds = []
    targets = []

    with torch.no_grad():

        for X,y in loader:

            X = X.to(device)

            output = model(X).cpu().numpy()

            preds.extend(output.flatten())

            targets.extend(y.numpy().flatten())

    r2 = r2_score(targets,preds)

    return r2


def predict(model,loader):
    pass


def save_artifacts(model,metrics):
    pass


def main():

    train_loader,test_loader = make_dataloaders()

    model = build_model()

    train(model,train_loader)

    r2 = evaluate(model,test_loader)

    print("R2:",r2)

    if r2 > 0.70:
        print("PASS")
        return 0
    else:
        print("FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())