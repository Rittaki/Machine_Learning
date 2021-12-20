import sys
import torch
import numpy as np
import torchvision
from torch.utils import data
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F

class MyDataSet(data.Dataset):
    def __init__(self, x, y=None, transform=None):
        self.__x = x
        self.__y = y
        self.__transform = transform

    def __len__(self):
        return len(self.__x)

    def __getitem__(self, index):
        x = self.__x[index]
        x = np.asarray(x).astype(np.uint8).reshape(28, 28)
        if self.__transform:
            x = self.__transform(x)
        if self.__y is not None:
            y = self.__y[index]
            return x, y
        return x

class ModelA(nn.Module):
    def __init__(self,image_size):
        super(ModelA, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def split(train_x, train_y):
    n = int(train_x.shape[0] * 0.2)
    x_valid = train_x[0:n, :]
    x_data = train_x[n:, :]
    y_valid = train_y[0:n]
    y_data = train_y[n:]
    return x_data, y_data, x_valid, y_valid

def main():
    train_x = np.loadtxt(sys.argv[1])
    train_y = np.loadtxt(sys.argv[2], dtype="int64")
    test_x = np.loadtxt(sys.argv[3])
    x_data, y_data, x_val, y_val = split(train_x, train_y)
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_data = MyDataSet(x_data, y_data, transforms)
    validation_data = MyDataSet(x_val, y_val, transforms)
    test_data = MyDataSet(test_x, None, transforms)
    # print(len(train_data))
    # print(train_data[1000])
    # print(len(test_data))
    # print(test_data[1000])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data)
    model_a = ModelA(image_size=28*28)

    lr = 0.1
    optimizer = torch.optim.SGD(model_a.parameters(), lr=lr)

    for epoch in range(1, 10 + 1):
        train(epoch, model_a, train_loader, optimizer)

    prediction = test(model_a, test_loader)
    write_to_file(prediction)

def train(epoch, model, train_loader, optimizer):
    model.train()
    losses = 0
    accuracy_counter = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        losses += F.nll_loss(output, labels, reduction="mean").item()
        loss.backward()
        optimizer.step()
        prediction = output.max(1, keepdim=True)[1]
        accuracy_counter += prediction.eq(labels.view_as(prediction)).cpu().sum()


    # Get the average loss
    losses /= (len(train_loader) * 64)
    print('\nTrain Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        losses, accuracy_counter, (len(train_loader) * 64),
        100. * accuracy_counter / (len(train_loader) * 64)))

# def (model, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             output = model(data)
#             test_loss += F.nll_loss(output, target, size_average=False).item()
#             pred = output.max(1, keepdim=True)[1]
#             correct += pred.eq(target.view_as(pred)).cpu().sum()
#     test_loss /= len(test_loader.dataset)
#     print('\nTestset: Average loss: {:.4f}, Accuracy: {} / {}({:.0f} % )\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

def test(model, test_loader):
    model.eval()
    test_y_list = []
    for data in test_loader:
        output = model(data)
        predict = output.max(1, keepdim=True)[1]
        test_y_list.append(str(int(predict)))
    return test_y_list

def write_to_file(predict_y):
    with open('test_y', 'w') as file:
        for y in predict_y:
            file.write("%s\n" % y)
    file.close()

if __name__ == '__main__':
    main()