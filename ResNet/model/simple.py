import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16*28*28, 10)
        # self.flatten = nn.Flatten()
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(28*28, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 10)
        # )


    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)

        # x = self.flatten(x)
        # x = self.linear_relu_stack(x)
        return x
    


        