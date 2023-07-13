import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.MNIST import MNIST
from model.simple import SimpleNet
from matplotlib import pyplot as plt

train_set = MNIST("data/train.csv")
train_loader = DataLoader(train_set, batch_size=32)

model = SimpleNet()
loss_fn = nn.CrossEntropyLoss()
optimzer = torch.optim.AdamW(model.parameters(), lr=1e-3)

def train():

    for batch, (imgs, labels) in enumerate(train_loader):
        size = len(train_set)
        
      
        # print(imgs.shape, labels.shape)
        
        labels_pred = model(imgs)

        
        loss = loss_fn(labels_pred, labels)

        
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(imgs)
            print(f"loss:{loss:>7f}, [{current:>5d}/{size:>5d}]")








train()