这里本来是存放Cifar_10的数据集,但文件太大,故不上传
``` python
train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=transforms,
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=transforms,
                                         download=True)
```
train.py的这两行代码会自动下载
