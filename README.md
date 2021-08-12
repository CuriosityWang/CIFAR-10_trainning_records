# CIFAR-10_trainning_records
使用了自己搭建的模型VGG_simplified,Vgg16,Resnet18对Cifar_10进行训练
## 具体效果
1. 各模型对比

|       模型       | 验证集准确率 | 参数数量(运算速度) |
| :--------------: | :----------: | :----------------: |
|       Cir        |  **66.58%**  |       90,080       |
|      Vgg16       |   **72%**    |    138,357,544     |
| Vgg16_simplified |   **76%**    |      573,888       |
|     Resnet18     |  **82.32%**  |     33,161,024     |

## 使用方法
1. 下载文件到本地
![image](https://user-images.githubusercontent.com/50990182/129144452-ad0bf580-90c9-4c3d-b8c0-2053868c6b89.png)
2. 运行train.py
![image](https://user-images.githubusercontent.com/50990182/129144616-a1c5d61b-2459-49a4-93cf-9cac266999f3.png)
可以把Cir_module修改为你想要测试的模型,模型在module.py中
3. 使用Resnet模型时可以导入训练好的Cir_9.pth,以减少训练时间.Cir_9.pth在releases/module_file下

训练笔记请看: https://www.cnblogs.com/programmerwang/p/15129658.html


