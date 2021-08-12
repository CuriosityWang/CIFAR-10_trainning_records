import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from module import *
import matplotlib.pyplot as plt
import torchvision.models as models

from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 500
name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

Cir_module = ResNet()
Cir_module = torch.load("cir_9.pth")
Cir_module.to(device)

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=transform,
                                         download=True)
test_dataloader = DataLoader(test_data, batch_size=500)

# 计算 val_acc
total_accuracy_nums = 0
Cir_module.eval()
with torch.no_grad():
    for data in test_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = Cir_module(imgs)
        nums = (outputs.argmax(1) == targets).sum()
        total_accuracy_nums = total_accuracy_nums + nums

val_acc = total_accuracy_nums / 10000
print(val_acc )