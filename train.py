import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np
from module import *

# 记录 test_acc train_acc
path = "train_out.txt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transforms = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])

train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=transforms,
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=transforms,
                                         download=True)

# 训练集 测试集大小
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64)

# 定义模型-ResNet
Cir_module = ResNet()
# model = torch.load("net.pth")
Cir_module.to(device)

# 定义优化器
optimizer = optim.Adam(Cir_module.parameters(), 0.001)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0

best_acc = 0
best_module = 0

train_acc = []
val_acc = []

for i in range(30):
    print("-------第 {} 轮训练开始-------".format(i + 1))

    f = open(path, "a+")
    # 训练步骤开始
    Cir_module.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = Cir_module(imgs)
        loss = loss_fn(outputs, targets)

        # 梯度更新

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1

        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
      
    # 测试步骤开始
    Cir_module.eval()
    total_test_loss = 0
    total_accuracy = 0

    total_train_acc_nums = 0

    # 计算 val_acc
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = Cir_module(imgs)
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    val_acc_epoch = total_accuracy / test_data_size
    val_acc.append(val_acc_epoch.item())
    print("Epoch_num: ", i, 'val_acc:', val_acc_epoch, file=f)
    print("Epoch_num: ", i, 'val_acc:', val_acc_epoch)
    # 计算 train_acc
    # 只计算10000个
    with torch.no_grad():
        train_acc_step = 0
        for data in train_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = Cir_module(imgs)
            accuracy_nums = (outputs.argmax(1) == targets).sum()
            total_train_acc_nums = total_train_acc_nums + accuracy_nums
            train_acc_step = train_acc_step + 1
            if train_acc_step == 156:
                break

    train_acc_epoch = total_train_acc_nums / 10048
    train_acc.append(train_acc_epoch.item())
    print("Epoch_num: ", i, 'train_acc:', train_acc_epoch, file=f)
    print("Epoch_num: ", i, 'train_acc:', train_acc_epoch)
    if (val_acc_epoch > best_acc):
        best_acc = val_acc_epoch
        best_module = i
    f.close()
    torch.save(Cir_module, "cir_{}.pth".format(i))
    print("模型已保存")

print(best_acc.item())
print("对应的module是cir_{}.pth".format(best_module))