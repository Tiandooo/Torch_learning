import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.MNIST import MNIST
from data.MNISTT import MNISTT
from model.simple import SimpleNet
from model.FirstNN import FirstNN
from matplotlib import pyplot as plt
import csv
batch_size = 32
train_set = MNIST("data/train.csv")
test_set = MNISTT("data/test.csv")
train_loader = DataLoader(train_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)
epoch = 5

model = SimpleNet()
loss_fn = nn.CrossEntropyLoss()
optimzer = torch.optim.AdamW(model.parameters(), lr=1e-4)

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




# for i in range(epoch):
#     train()


# torch.save(model.state_dict(), 'model_parameters.pth')

model.load_state_dict(torch.load("model_parameters.pth"))
# with open("res.txt", "a") as f:
#     for i, imgs in enumerate(test_loader):
#         y = model(imgs)
#         blabel = y.argmax(1)
#         blabel = blabel.tolist()
        
        
#         for j, label in enumerate(blabel):
            
#             index = i * batch_size + j + 1
#             # print(index, l)
            
#             f.write("%d,%d\n" % (index, label))
        
with open('data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ImageId", "Label"])
    for i, imgs in enumerate(test_loader):
        y = model(imgs)
        blabel = y.argmax(1)
        blabel = blabel.tolist()
        
        
        for j, label in enumerate(blabel):
            
            index = i * batch_size + j + 1
            # print(index, l)
            writer.writerow([index, label])
    



