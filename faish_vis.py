import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import cv2
from matplotlib import pyplot as plt
import os
from custom import CustomDataset

cust_data = CustomDataset(
    label="data/labels.txt",
    img_path="data/img"
)

training_data = datasets.FashionMNIST(
    root="data",
    download=False,
    train=True,
    transform=ToTensor(),
)

testing_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor()
)

train_loader = DataLoader(training_data, batch_size=1)
test_loader = DataLoader(testing_data, batch_size=1)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

num_row = 3
num_col = 5

num = num_row * num_col

figure = plt.figure(figsize=(8, 8))

# 展示一个
# for X,y in train_loader:
#     # squeeze()去除长度为1的dim，cmap设置灰度图
#     plt.imshow(X.squeeze(), cmap='gray')
#     plt.show()
#     print(X,y)
#     break
img_path = "data/img"
if not os.path.exists(img_path):
    os.mkdir(img_path)

for i in range(1, num_row * num_col + 1):
    sample_idx = torch.randint(len(cust_data), size=(1,)).item()
    img, label = cust_data[sample_idx]
    
    figure.add_subplot(num_row, num_col, i)
    plt.title(labels_map[label])
    plt.axis("off")

    # 方式1，# C x H x W  ---> H x W x C
    img = img.transpose(0, 2).transpose(0, 1)
    plt.imshow(img)
    # 方式2，squeeze
    # plt.imshow(img.squeeze(), cmap="gray")


    # 保存图片和标签
    # filename = os.path.join(img_path,"%04d" % i + ".jpg")
    # print(filename)
    # plt.imsave(filename, img.squeeze(),cmap="gray")
    # with open("data/labels.txt", mode="a") as f:
    #     f.write("%s\n" % label)
plt.show()