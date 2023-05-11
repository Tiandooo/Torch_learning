import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    download=True,
    train=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    download=False,
    root="data",
    train=False,
    transform=ToTensor(),
)

train_loader = DataLoader(training_data, batch_size=1)
test_loader = DataLoader(test_data, batch_size=1)

for X, y in train_loader:
    print(f"Shape of X[N, C, H, W] {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
