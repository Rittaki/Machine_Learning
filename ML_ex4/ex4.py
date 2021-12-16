import sys
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class MyData(Dataset):
    def __init__(self, x, y=None, transforms=None):
        self.data_x = x
        self.data_y = y
        self.transforms = transforms

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, item):
        x = self.data_x[item]
        x = np.asarray(x).astype(np.uint8).reshape(28, 28)
        if self.transforms:
            x = self.transforms(x)
        if self.data_y is not None:
            return x, self.data_y[item]
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
    train_y = np.loadtxt(sys.argv[2])
    test_x = np.loadtxt(sys.argv[3])
    x_data, y_data, x_val, y_val = split(train_x, train_y)
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_data = MyData(x_data, y_data, transforms)
    validation_data = MyData(x_val, y_val, transforms)
    test_data = MyData(test_x, None, transforms)
    # print(len(train_data))
    # print(train_data[1000])
    # print(len(test_data))
    # print(test_data[1000])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
    model_a = ModelA(image_size=28*28)

    lr = 0.1
    optimizer = torch.optim.SGD(model_a.parameters(), lr=lr)

    for epoch in range(1, 10 + 1):
        train(epoch, model_a, train_loader, optimizer)

def train(epoch, model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()