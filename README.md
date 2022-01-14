# README

### 目录结构

目录结构及对应内容如下所示：

```
├─code
│  ├─basic_model			基本模型,如alexnet,lenet等
│  ├─official_example       给定的样例模型
│  └─resnet 				resnet相关模型
├─docs 						报告
|-image						输出的部分示意图
└─result
    ├─basic_model			基本模型结果
    └─resnet				resnet相关模型结果
```





### 模型概述

我们提供两种模型以供审阅，预计完成时间的运行环境使用一块RTX3080 Ti测出。

| 模型名称  | 模型概述                              | 准确度 | 建议采用的Epochs数量 |
| --------- | ------------------------------------- | ------ | -------------------- |
| ResNet18  | 在保证准确率高于90%的前提下，更快完成 | 90.18% | 100                  |
| ResNet101 | 更高准确率，但是运行时间可能更长      | 92.00% | 100                  |



### Run ResNet18

在最外层目录，`python ./cifar10_resnet_pytorch/trainer.py res18`即可运行。

### Run ResNet101

在最外层目录，`python ./cifar10_resnet_pytorch/trainer.py res101`即可运行。



### 模型结果

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

