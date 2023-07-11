from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from MNIST import MNIST

train_mnist = MNIST("data/train.csv")
train_loader = DataLoader(train_mnist, batch_size=5)

for i in train_loader:
    imgs = i[0]
    labels = i[0]
    
    plt.imshow(imgs[3])
    plt.show()
    
    break