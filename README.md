# README

我们提供两种模型以供审阅，预计完成时间的运行环境使用一块RTX3080 Ti测出。

| Name      | Description                           | Esi-running-time | Esi-Acc | Adviced Epochs |
| --------- | ------------------------------------- | ---------------- | ------- | -------------- |
| ResNet18  | 在保证准确率高于90%的前提下，更快完成 |                  | 90.18%  | 100            |
| ResNet101 | 更高准确率，但是运行时间可能更长      |                  | 92%     | 100            |

## Run ResNet18

在最外层目录，`python ./cifar10_resnet_pytorch/trainer.py res18`即可运行。

## Run ResNet101

在最外层目录，`python ./cifar10_resnet_pytorch/trainer.py res101`即可运行。