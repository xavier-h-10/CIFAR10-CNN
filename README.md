# CIFAR-10 Training Based on ResNet

### 概述

我们主要采用ResNet以及一些基础的模型，对CIFAR-10进行训练。

### 目录结构

目录结构及对应内容如下所示：

```
├─code
│  ├─basic_model			基本模型,如AlexNet,Inception等
│  ├─official_example       给定的样例模型
│  └─resnet					ResNet相关模型
│          lenet.py			使用LetNet作为基准进行比较
│          plot.py			作图用	
│          resnet.py	    定义了ResNet-18模型
│          resnet_all.py	定义了ResNet-50以及更深的模型
│          train.py			训练主要代码
├─docs 						报告
|-image						输出的部分示意图(详见文档)
└─result
    ├─basic_model			基本模型结果
    └─resnet				ResNet相关模型结果
    	├─accuracy			每一轮训练后的准确度
        └─log				每一轮训练产生的log
```

### 模型说明

ResNet相关模型的结果存放在`./result/resnet`内，由于命名有一些复杂，因此此处进行详细说明。

| 文件名称               | 网络层数 | batch大小 | 学习率大小 | 说明                  |
| ---------------------- | -------- | --------- | ---------- | --------------------- |
| resnet18_batch8        | 18       | 8         | 8e-3       |                       |
| resnet18_batch32       | 18       | 32        | 8e-3       |                       |
| resnet18_batch64       | 18       | 64        | 8e-3       |                       |
| resnet18_batch128_8e-1 | 18       | 128       | 8e-1       |                       |
| resnet18_batch128_8e-2 | 18       | 128       | 8e-2       |                       |
| resnet18_batch128_8e-3 | 18       | 128       | 8e-3       |                       |
| resnet18_batch128_8e-4 | 18       | 128       | 8e-4       |                       |
| resnet50_2gpu_batch256 | 50       | 256       | 8e-3       | 使用了两块GPU并行训练 |
| resnet50_with_amp      | 50       | 256       | 8e-3       | 使用了数据增强        |
| resnet101_batch128     | 101      | 128       | 8e-3       |                       |
| lenet_batch128         | 5        | 128       | 8e-3       | LeNet模型，对比用     |

### 代码运行

- 对于ResNet相关模型，运行``./code/resnet/train.py``即可，可根据注释调整模型层数、学习率等参数。
- 对于基本模型，运行``./code/basic_model``下对应的文件即可。
- 需要特别注意，由于我们进行了分工，因此ResNet相关模型采用了PyTorch进行训练，而基本模型采用了TensorFlow进行训练。
