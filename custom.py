import torch
from torch import nn 
from torch.utils.data import dataloader,Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from matplotlib import pyplot as plt
import os

class CustomDataset(Dataset):
    def __init__(self, label, img_path, transform=None, target_transform=None) -> None:
        with open(label, mode="r") as f:
            labels = [int(i) for i in f.readlines()]
        self.labels = labels
        self.img_path = img_path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img = os.path.join(self.img_path, "%04d.jpg" % index)
        img = read_image(img)
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            img = self.target_transform(img)
        
        return img, label
