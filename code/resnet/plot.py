import os

import matplotlib.pyplot as plt
import re

testdir = r'./result/test'

traindir = r'./result/train'

train_savedir_loss = r'./result/output/train_loss.png'
train_savedir_acc = r'./result/output/train_acc.png'
test_savedir = r'./result/output/test.png'


def read_test_data(filepath):
    with open(filepath, 'r') as f:
        y = []
        lines = f.readlines()
        for line in lines:
            number = float(re.findall(r'\d+\.\d+', line)[0])
            y.append(number)
        x = [i for i in range(len(lines))]
        return x, y


def read_train_data(filepath):
    with open(filepath, 'r') as f:
        loss = []
        acc = []
        lines = f.readlines()
        for line in lines:
            loss_and_acc = re.findall(r'\d+\.\d+', line)
            loss.append(float(loss_and_acc[0]))
            acc.append(float(loss_and_acc[1]))
        x = [i for i in range(len(lines))]
        return x, loss, acc


def plot_all(output, xs, ys, labels, suffix=''):
    min_len = len(xs[0])
    for i in range(len(xs)):
        min_len = min(min_len, len(xs[i]))
    for i in range(len(xs)):
        x, y, label = xs[i][0:min_len], ys[i][0:min_len], labels[i] + suffix
        plt.plot(x, y, label=label)
    plt.legend()
    plt.savefig(output, dpi=1000)
    plt.show()
    plt.cla()


def plot_train():
    xs = []
    losses = []
    accs = []
    labels = []
    for _, _, files in os.walk(traindir):
        for file in files:
            abs_path = traindir + '/' + file
            x, loss, acc = read_train_data(abs_path)
            label = file.replace('log_', '').replace('.txt', '')
            xs.append(x)
            losses.append(loss)
            accs.append(acc)
            labels.append(label)
    plot_all(train_savedir_loss, xs, losses, labels, '_loss')
    plot_all(train_savedir_acc, xs, accs, labels, '_acc')


def plot_test():
    xs = []
    accs = []
    labels = []
    for _, _, files in os.walk(testdir):
        for file in files:
            abs_path = testdir + '/' + file
            x, acc = read_test_data(abs_path)
            label = file.replace('acc_', '').replace('.txt', '')
            xs.append(x)
            accs.append(acc)
            labels.append(label)
    plot_all(test_savedir, xs, accs, labels, '_acc')


def main():
    plot_train()
    plot_test()


if __name__ == '__main__':
    main()
