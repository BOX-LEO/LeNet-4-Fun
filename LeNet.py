import numpy as np
import pylab as p
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        # first convolution layer take RGB image and output 6 features
        self.conv1 = nn.Conv2d(3, 6, 5)
        # second convolution layer take 6 channel and output 16 features
        self.conv2 = nn.Conv2d(6, 16, 5)
        # pooling layers are max pooling
        self.maxpool = nn.MaxPool2d(2)
        # Full connect layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x should be RGB image
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5_norm(nn.Module):
    def __init__(self):
        super(LeNet5_norm, self).__init__()
        # first convolution layer take RGB image and output 6 features
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.norm1 = nn.BatchNorm2d(6)
        # second convolution layer take 6 channel and output 16 features
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.norm2 = nn.BatchNorm2d(16)
        # pooling layers are max pooling
        self.maxpool = nn.MaxPool2d(2)
        # Full connect layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x should be RGB image
        x = self.maxpool(F.relu(self.norm1(self.conv1(x))))
        x = self.maxpool(F.relu(self.norm2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def loadData(path, bs=128):
    # calculate mean and std of dataset
    pre_transformer = transforms.Compose([transforms.Resize((32, 32)),
                                          transforms.ToTensor()])
    pre_dataset = datasets.ImageFolder(path, transform=pre_transformer)
    lorder = torch.utils.data.DataLoader(pre_dataset, batch_size=pre_dataset.__len__())
    data = next(iter(lorder))[0]
    mean, std = torch.mean(data, (0, 2, 3)), torch.std(data, (0, 2, 3))

    # dataset normalization
    data_transformer = transforms.Compose([transforms.Resize((32, 32)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=mean, std=std)])
    dataset = datasets.ImageFolder(path, transform=data_transformer)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)

    return data_loader


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    # model setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = LeNet5()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    adam = optim.Adam(net.parameters())
    batchsize = 128
    l2_lambda = 0.001
    regularization = True
    print('model set')
    # load training data
    train_loader = loadData('train/img', batchsize)
    val_loader = loadData('val/img', batchsize)
    # test_loader = loadData('test/img', batchsize)
    print('data loaded')

    # training loop:
    loss_list = []
    val_loss_list = []
    for epoch in range(90):
        print('epoch=', epoch)
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            image, label = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            adam.zero_grad()
            # forward + backward + optimize
            output = net(image)
            l2_norm = sum(p.pow(2.0).sum() for p in net.parameters())
            if regularization:
                loss = criterion(output, label) + l2_lambda * l2_norm
            else:
                loss = criterion(output, label)
            loss.backward()
            adam.step()

            loss_list.append(loss.item())
            # print(loss.item(), epoch)

        # save validation result
        if epoch % 5 == 4:
            val_loss = 0.0
            for i, data in enumerate(val_loader, 0):
                val_im, val_lab = data[0].to(device), data[1].to(device)
                val_out = net(val_im)
                loss = criterion(val_out, val_lab)
                val_loss += loss.item()
            val_loss_list.append(val_loss)
    PATH = './LeNet5_reg.pth'
    torch.save(net.state_dict(), PATH)

    plt.plot(list(range(len(loss_list))), loss_list)
    plt.title('training loss')
    plt.show()
    plt.plot(list(range(len(val_loss_list))), val_loss_list)
    plt.title('validation loss every 5 epochs')
    plt.show()
