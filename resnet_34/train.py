import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import ResNet18
from resnet_all import ResNet34
import numpy as np
import random
import ssl
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def imshow(img):
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_some_data(dataset, num=1):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False,
                                             num_workers=0)
    cnt = 0
    for inputs, labels in dataloader:
        if cnt >= num:
            return
        imshow(torchvision.utils.make_grid(inputs))
        cnt += 1


setup_seed(2021)

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0))

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')  # 输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  # 恢复训练时的模型路径
args = parser.parse_args()

# 超参数设置
EPOCH = 135  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128  # 批处理尺寸(batch_size)
LR = 0.1  # 学习率
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

# 准备数据集并预处理
transform_train_amplified1 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，再把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),  # R,G,B每层的归一化用到的均值和方差
])

amplified_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                  transform=transform_train_amplified1)
show_some_data(amplified_trainset)

transform_train_amplified2 = transforms.Compose([
    transforms.RandomAffine(degrees=20),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),  # R,G,B每层的归一化用到的均值和方差
])

amplified_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                  transform=transform_train_amplified2)
show_some_data(amplified_trainset)

transform_train_amplified3 = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),  # R,G,B每层的归一化用到的均值和方差
])

amplified_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                  transform=transform_train_amplified3)
show_some_data(amplified_trainset)

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),  # R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
trainset_amplied1 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                 transform=transform_train_amplified1)
trainset_amplied2 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                 transform=transform_train_amplified2)
trainset_amplied3 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                 transform=transform_train_amplified3)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)  # 训练数据集

trainset = trainset.__add__(trainset_amplied1)
trainset = trainset.__add__(trainset_amplied2)
trainset = trainset.__add__(trainset_amplied3)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# net = ResNet18().to(device)
net = ResNet18().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                      weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# 训练
if __name__ == "__main__":
    best_acc = 85  # 初始化best test accuracy
    with open("acc.txt", "w") as acc_file:
        with open("log.txt", "w") as log_file:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # preparation
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    log_file.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                                   % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    log_file.write('\n')
                    log_file.flush()

                print("Starting test...")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    print('Saving model......')
                    # torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    acc_file.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    acc_file.write('\n')
                    acc_file.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        with open("best_acc.txt", "w") as f:
                            f.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                            f.close()
                            best_acc = acc
                scheduler.step()

            print("Training Finished, TotalEPOCH=%d" % EPOCH)
