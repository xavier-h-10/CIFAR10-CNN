import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from model import Net
import ssl
import random
import time


def show_kernel(model):
    # 可视化卷积核
    for name, param in model.named_parameters():
        torch.no_grad()
        if 'conv' in name and 'weight' in name:
            with SummaryWriter(comment='model') as w:
                in_channels = param.size()[1]
                out_channels = param.size()[0]  # 输出通道，表示卷积核的个数
                k_w, k_h = param.size()[3], param.size()[2]  # 卷积核的尺寸
                kernel_all = param.view(-1, 1, k_w, k_h)  # 每个通道的卷积核
                # print(kernel_all)
                kernel_grid = torchvision.utils.make_grid(kernel_all)
                w.add_image(f'{name}_all', kernel_grid, global_step=0)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    ssl._create_default_https_context = ssl._create_unverified_context

    setup_seed(2021)

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.current_device())

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    train_on_gpu = True
    # functions to show an image
    device = torch.device("cuda:0" if train_on_gpu else "cpu")

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    dummy_input = torch.rand(batch_size, 3, 32, 32).to(device)

    with SummaryWriter(comment='model') as w:
        w.add_graph(net, dummy_input)

        start_time = time.time()

        for epoch in range(100):  # loop over the dataset multiple times
            print("start training...")
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # print("Train data...")

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    end_time = time.time()
                    print('[%d, %5d] loss: %.3f training_time: %.6f s' %
                          (epoch + 1, i + 1, running_loss / 2000, end_time - start_time))
                    running_loss = 0.0
                    w.add_scalar('loss', running_loss, epoch)
                    start_time = end_time
                    show_kernel(net)

    print('Finished Training')
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)


if __name__ == "__main__":
    main()
