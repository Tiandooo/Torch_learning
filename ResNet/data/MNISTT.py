import torch
from torch import nn
from torch.utils.data import dataloader, Dataset
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
class MNISTT(Dataset):
    def __init__(self, file_path:str, transform=None) -> None:
        self.file_path = file_path
        

        self.imgs = pd.read_csv(self.file_path).values
        self.imgs = torch.tensor(self.imgs, dtype=torch.float32)
        self.imgs = torch.unsqueeze(self.imgs, dim=1)
        
        self.transform = transform

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img = self.imgs[index].reshape([1, 28, 28])
        # print(label, img)
        if self.transform:
            img = self.transform(img)
        return img
    
if __name__ == "__main__":
    mnist = MNISTT("test.csv")
    mnist.__getitem__(0)