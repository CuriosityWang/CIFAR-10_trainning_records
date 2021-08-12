import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np
from module import *
import matplotlib.pyplot as plt
import torchvision.models as models

from PIL import Image

batch_size = 500
name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model = ResNet()
model = torch.load("cir_9.pth")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


im = Image.open('./test_data/test.png')
plt.imshow(im)
plt.show()
im = transform(im)
print(im.shape)
im = im.type(torch.cuda.FloatTensor)
im = torch.reshape(im, (1, 3, 224, 224))
print(im.shape)
predict = model(im)
pred = predict.argmax(dim=1)
print(name[pred])
