import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
class FirstNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # 3 x 28 x 28 => 3 x 784

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        # print(x.size())
        logits = self.linear_relu_stack(x)
        return logits