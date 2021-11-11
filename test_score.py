import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision
from sklearn.metrics import confusion_matrix


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
    pre_transformer = transforms.Compose([transforms.Resize((32, 32)),
                                          transforms.ToTensor()])
    pre_dataset = datasets.ImageFolder(path, transform=pre_transformer)
    lorder = torch.utils.data.DataLoader(pre_dataset, batch_size=pre_dataset.__len__())
    data = next(iter(lorder))[0]
    mean, std = torch.mean(data, (0, 2, 3)), torch.std(data, (0, 2, 3))

    data_transformer = transforms.Compose([transforms.Resize((32, 32)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=mean, std=std)])
    dataset = datasets.ImageFolder(path, transform=data_transformer)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs)

    # images = next(iter(data_loader))[0]
    # imshow(torchvision.utils.make_grid(images))
    return data_loader


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batchsize = 1024
    net = LeNet5()
    net.load_state_dict(torch.load('./LeNet5_reg.pth'))
    net.to(device)
    test_loader = loadData('test/img', batchsize)
    print('data loaded')
    correct = 0
    total = 0
    pred = []
    labe = []
    incorrect_img=[]
    incorrect_label = []
    for data in test_loader:
        image, label = data[0].to(device), data[1].to(device)
        output = net(image)
        _, predicted = torch.max(output.data, 1)
        labe.append(label.cpu().numpy().tolist())
        total += label.size(0)
        correct += (predicted == label).sum().item()
        # save prediction result
        pred.append(predicted.cpu().numpy().tolist())
        # get misclassified images
        mis_index = ((predicted == label) == False)
        incorrect_img.append(image[mis_index].cpu())
        incorrect_label.append(predicted[mis_index].cpu().numpy())
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    prediction = [p for sublist in pred for p in sublist]
    labels = [l for sublist in labe for l in sublist]
    cof = confusion_matrix(labels, prediction)
    acc = []
    for i in range(10):
        acc.append(cof[i][i]/sum(cof[i]))
    print(cof)
    print(acc)
    print(correct / total)
    imshow(torchvision.utils.make_grid(incorrect_img[0][:16]))
    print(' '.join('%5s' % classes[incorrect_label[0][j]] for j in range(16)))

