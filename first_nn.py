import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from model.FirstNN import FirstNN
device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")
# hyper parameter
batch_size = 32
learning_rate = 1e-3
epochs = 5

train_data = datasets.FashionMNIST(
    root="data",
    download=False,
    transform=ToTensor(),
    train=True
)

test_data = datasets.FashionMNIST(
    root="data",
    download=False,
    transform=ToTensor(),
    train=False
)

train_loader = DataLoader(train_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

     
model = FirstNN().to(device)
# print(model)

# X = torch.rand(3, 28, 28, device=device)
# y = model(X)

def train(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    for batch, (X, y) in enumerate(data_loader):
        print(X.shape, y.shape, X.dtype, y.dtype)
        
        y_pred = model(X)
        # print(y)
        # print(y_pred)
        loss = loss_fn(y_pred, y)  # 顺序不能反

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(data_loader, model, loss_fn):
    size = len(data_loader.dataset)
    num_batchs = len(data_loader)
    test_loss, correct = 0.0, 0

    for X, y in data_loader:
        y_pred = model(X)
        test_loss += loss_fn(y_pred, y).item()  # 顺序不能反
        correct += (y_pred.argmax(1)==y).type(torch.float).sum().item()

    test_loss /= num_batchs
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss:{test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for e in range(epochs):
    print(f"Epoch {e+1}\n-------------------------")
    train(train_loader, model, loss_fn, optimizer)
    # test(test_loader, model, loss_fn)
print("Saving model... \n")
torch.save(model.state_dict(), 'model1_weights.pth')
print("Done!")