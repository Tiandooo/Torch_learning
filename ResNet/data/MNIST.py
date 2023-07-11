import torch
from torch import nn
from torch.utils.data import dataloader, Dataset
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
class MNIST(Dataset):
    def __init__(self, file_path:str, transform=None) -> None:
        self.file_path = file_path
        self.labels = pd.read_csv(self.file_path, usecols=[0]).values
        self.imgs = pd.read_csv(self.file_path, usecols=list(range(1, 785))).values
        
       
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = self.labels[index]
        img = self.imgs[index].reshape([28, 28])
        # print(label, img)
        if self.transform:
            img = self.transform(img)
        return img, label
    
if __name__ == "__main__":
    mnist = MNIST("data/train.csv")
    mnist.__getitem__(0)