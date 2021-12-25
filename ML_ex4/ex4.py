import sys
import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F

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

class ModelB(nn.Module):
    def __init__(self,image_size):
        super(ModelB, self).__init__()
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

class ModelC(nn.Module):
    def __init__(self,image_size):
        super(ModelC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ModelD(nn.Module):
    def __init__(self,image_size):
        super(ModelD, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc0_bn = nn.BatchNorm1d(100)
        self.fc1 = nn.Linear(100, 50)
        self.fc1_bn = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0_bn(self.fc0(x)))
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ModelE(nn.Module):
    def __init__(self,image_size):
        super(ModelE, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)

class ModelF(nn.Module):
    def __init__(self,image_size):
        super(ModelF, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)

class BestModel(nn.Module):
    def __init__(self,image_size):
        super(BestModel, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 256)
        self.fc0_bn = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc2_bn = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        # x = F.relu(self.fc0(x))
        x = F.relu(self.fc0_bn(self.fc0(x)))
        x = self.dropout(x)
        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
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
    train_data = MyData(x_data, y_data, transforms)
    validation_data = MyData(x_val, y_val, transforms)
    test_data = MyData(test_x, None, transforms)
    test_set_orig = torchvision.datasets.FashionMNIST(root='./data',
                                                 train=False,
                                                 download=True,
                                                 transform=transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data)
    orig_test_loader = torch.utils.data.DataLoader(test_set_orig,
                                              batch_size=64,
                                              shuffle=False)
    """
    lr = 0.1
    model_a = ModelA(image_size=28*28)
    optimizer_sgd = torch.optim.SGD(model_a.parameters(), lr=lr)

    train_loss_list = list()
    train_accuracy_list = list()
    valid_loss_list = list()
    valid_accuracy_list = list()

    for epoch in range(1, 10 + 1):
        loss, accuracy = train(epoch, model_a, train_loader, optimizer_sgd)
        train_loss_list.append(loss)
        train_accuracy_list.append(accuracy)
        valid_loss, valid_accuracy = valid(model_a, valid_loader)
        valid_loss_list.append(valid_loss)
        valid_accuracy_list.append(valid_accuracy)

        test_orig(model_a, orig_test_loader)

    prediction = test(model_a, test_loader)
    """
    """
    lr = 0.001
    model_b = ModelB(image_size=28*28)
    optimizer_adam = torch.optim.Adam(model_b.parameters(), lr=lr)

    train_loss_list = list()
    train_accuracy_list = list()
    valid_loss_list = list()
    valid_accuracy_list = list()

    for epoch in range(1, 10 + 1):
        loss, accuracy = train(epoch, model_b, train_loader, optimizer_adam)
        train_loss_list.append(loss)
        train_accuracy_list.append(accuracy)
        valid_loss, valid_accuracy = valid(model_b, valid_loader)
        valid_loss_list.append(valid_loss)
        valid_accuracy_list.append(valid_accuracy)

        test_orig(model_b, orig_test_loader)

    prediction = test(model_b, test_loader)
    """
    """
    lr = 0.001
    model_c = ModelC(image_size=28 * 28)
    optimizer_adam = torch.optim.Adam(model_c.parameters(), lr=lr)

    train_loss_list = list()
    train_accuracy_list = list()
    valid_loss_list = list()
    valid_accuracy_list = list()

    for epoch in range(1, 10 + 1):
        loss, accuracy = train(epoch, model_c, train_loader, optimizer_adam)
        train_loss_list.append(loss)
        train_accuracy_list.append(accuracy)
        valid_loss, valid_accuracy = valid(model_c, valid_loader)
        valid_loss_list.append(valid_loss)
        valid_accuracy_list.append(valid_accuracy)

        test_orig(model_c, orig_test_loader)

    prediction = test(model_c, test_loader)
    """
    """
    lr = 0.001
    model_d = ModelD(image_size=28 * 28)
    optimizer_adam = torch.optim.Adam(model_d.parameters(), lr=lr)

    train_loss_list = list()
    train_accuracy_list = list()
    valid_loss_list = list()
    valid_accuracy_list = list()

    for epoch in range(1, 10 + 1):
        loss, accuracy = train(epoch, model_d, train_loader, optimizer_adam)
        train_loss_list.append(loss)
        train_accuracy_list.append(accuracy)
        valid_loss, valid_accuracy = valid(model_d, valid_loader)
        valid_loss_list.append(valid_loss)
        valid_accuracy_list.append(valid_accuracy)

        test_orig(model_d, orig_test_loader)

    prediction = test(model_d, test_loader)
    """
    """
    lr = 0.001
    model_e = ModelE(image_size=28 * 28)
    optimizer_adam = torch.optim.Adam(model_e.parameters(), lr=lr)

    train_loss_list = list()
    train_accuracy_list = list()
    valid_loss_list = list()
    valid_accuracy_list = list()

    for epoch in range(1, 10 + 1):
        loss, accuracy = train(epoch, model_e, train_loader, optimizer_adam)
        train_loss_list.append(loss)
        train_accuracy_list.append(accuracy)
        valid_loss, valid_accuracy = valid(model_e, valid_loader)
        valid_loss_list.append(valid_loss)
        valid_accuracy_list.append(valid_accuracy)

        test_orig(model_e, orig_test_loader)

    prediction = test(model_e, test_loader)
    """
    """
    lr = 0.005
    model_f = ModelF(image_size=28 * 28)
    optimizer_adam = torch.optim.Adam(model_f.parameters(), lr=lr)

    train_loss_list = list()
    train_accuracy_list = list()
    valid_loss_list = list()
    valid_accuracy_list = list()

    for epoch in range(1, 10 + 1):
        loss, accuracy = train(epoch, model_f, train_loader, optimizer_adam)
        train_loss_list.append(loss)
        train_accuracy_list.append(accuracy)
        valid_loss, valid_accuracy = valid(model_f, valid_loader)
        valid_loss_list.append(valid_loss)
        valid_accuracy_list.append(valid_accuracy)

        test_orig(model_f, orig_test_loader)

    prediction = test(model_f, test_loader)
    """
    lr = 0.002
    model = BestModel(image_size=28 * 28)
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_list = list()
    train_accuracy_list = list()
    valid_loss_list = list()
    valid_accuracy_list = list()

    for epoch in range(1, 30 + 1):
        loss, accuracy = train(epoch, model, train_loader, optimizer_adam)
        train_loss_list.append(loss)
        train_accuracy_list.append(accuracy)
        valid_loss, valid_accuracy = valid(model, valid_loader)
        valid_loss_list.append(valid_loss)
        valid_accuracy_list.append(valid_accuracy)

        test_orig(model, orig_test_loader)

    prediction = test(model, test_loader)

    draw_loss(train_loss_list, valid_loss_list)
    draw_accuracy(train_accuracy_list, valid_accuracy_list)

    write_to_file(prediction)
    check_test()

def train(epoch, model, train_loader, optimizer):
    model.train()
    losses = 0
    accuracy_counter = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        losses += F.nll_loss(output, labels, reduction="sum").item()
        loss.backward()
        optimizer.step()
        prediction = output.max(1, keepdim=True)[1]
        accuracy_counter += prediction.eq(labels.view_as(prediction)).cpu().sum()

    losses /= (len(train_loader) * 64)
    print('\nEpoch: {}, Train Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
        losses, accuracy_counter, (len(train_loader) * 64),
        100. * accuracy_counter / (len(train_loader) * 64)))
    accuracy = 100. * accuracy_counter / (len(train_loader) * 64)
    return losses, accuracy

def valid(model, valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            output = model(data)
            valid_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    valid_loss /= len(valid_loader.dataset)
    print('\nValid Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    accuracy = 100. * correct / len(valid_loader.dataset)
    return valid_loss, accuracy

def test(model, test_loader):
    model.eval()
    test_y_list = []
    for data in test_loader:
        output = model(data)
        predict = output.max(1, keepdim=True)[1]
        test_y_list.append(str(int(predict)))
    return test_y_list

def test_orig(model, test_loader):
    prediction_list = []
    model.eval()
    test_loss = 0
    accuracy_counter = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            prediction = output.max(1, keepdim=True)[1]
            prediction_list.append(prediction)
            accuracy_counter += prediction.eq(target.view_as(prediction)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, accuracy_counter, len(test_loader.dataset), 100. * accuracy_counter / len(test_loader.dataset)))

def write_to_file(predict_y):
    with open('test_y', 'w') as file:
        for y in predict_y:
            file.write("%s\n" % y)
    file.close()

def check_test():
    my_y = np.loadtxt("./test_y", dtype="int64")
    true_y = np.loadtxt("./test_labels.txt", dtype="int64")
    array = np.where(my_y == true_y)
    accuracy_counter = len(array[0])
    print('\nTest Set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        accuracy_counter, (len(my_y)),
        100. * accuracy_counter / (len(my_y))))

def draw_loss(train_loss_values, valid_loss_values):
    epochs = range(1, len(train_loss_values) + 1)
    plt.plot(epochs, train_loss_values, label='Train')
    plt.plot(epochs, valid_loss_values, label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Graph")
    plt.legend()
    plt.show()

def draw_accuracy(train_accuracy, valid_accuracy):
    epochs = range(1, len(train_accuracy) + 1)
    plt.plot(epochs, train_accuracy, label='Train')
    plt.plot(epochs, valid_accuracy, label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Graph")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()