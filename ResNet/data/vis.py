from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from MNIST import MNIST
from MNISTT import MNISTT

train_mnist = MNIST("data/train.csv")
train_loader = DataLoader(train_mnist, batch_size=5)

test_set = MNISTT("data/test.csv")
test_loader = DataLoader(test_set, batch_size=5)
# for i in train_loader:
#     imgs = i[0]
#     labels = i[0]
    
#     plt.imshow(imgs[3])
#     plt.show()
    
#     break

for i in test_loader:
    for j in range(5):
        imgs = i[j]
        img = imgs.squeeze(0)
        plt.imshow(img)
        plt.show()
        
    break