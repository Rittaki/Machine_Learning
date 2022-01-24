import os
import torch
import torchvision
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
import gcommand_dataset
from gcommand_dataset import GCommandLoader

cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

trainData = GCommandLoader('./data/train')
validData = GCommandLoader('./data/valid')
testData = GCommandLoader('./data/test')
# trainData = GCommandLoader('./gcommands/train')
# validData = GCommandLoader('./gcommands/valid')
# testData = GCommandLoader('./gcommands/test')


classes = {0: "bed", 1: "bird", 2: "cat", 3: "dog", 4: "down", 5: "eight", 6: "five", 7: "four", 8: "go", 9: "happy",
           10: "house", 11: "left", 12: "marvin", 13: "nine", 14: "no", 15: "off", 16: "on", 17: "one", 18: "right",
           19: "seven", 20: "sheila", 21: "six", 22: "stop", 23: "three", 24: "tree", 25: "two", 26: "up", 27: "wow",
           28: "yes", 29: "zero"}


class CNNnetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(kernel_size=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(kernel_size=2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(kernel_size=2))
        self.linear = nn.Linear(256 * 10 * 6, 30)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)


def load_data():
    train_loader = torch.utils.data.DataLoader(
        trainData, batch_size=100, shuffle=True, num_workers=1, pin_memory=True, sampler=None)
    valid_loader = torch.utils.data.DataLoader(
        validData, batch_size=100, shuffle=False, num_workers=1, pin_memory=True, sampler=None)
    test_loader = torch.utils.data.DataLoader(
        testData, batch_size=100, shuffle=False, num_workers=1, pin_memory=True, sampler=None)
    return train_loader, valid_loader, test_loader


def train(epoch, model, trainLoader, optimizer, criterion):
    model.train()
    losses = 0
    accuracy_counter = 0
    size = trainLoader.dataset.len
    for idx, data in enumerate(trainLoader):
        images, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        losses += loss.item()
        prediction = output.max(1, keepdim=True)[1]
        accuracy_counter += prediction.eq(labels.view_as(prediction)).cpu().sum()

    losses /= len(trainLoader)
    # print('\nEpoch: {}, Train Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     epoch, losses, accuracy_counter, size, 100. * accuracy_counter / size))
    accuracy = 100. * accuracy_counter / size
    return losses, accuracy


def valid(model, valid_loader, criterion):
    model.eval()
    losses = 0
    correct = 0
    size = valid_loader.dataset.len
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            images, labels = data[0].to(device), data[1].to(device)
            output = model(images)
            valid_loss = criterion(output, labels)
            losses += valid_loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).cpu().sum()
    losses /= len(valid_loader)
    # print('\nValid Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     losses, correct, size, 100. * correct / size))
    accuracy = 100. * correct / size
    return losses, accuracy


def test(model, test_loader):
    model.eval()
    files_name = test_loader.dataset.spects
    test_y_list = []
    i = 0
    for data in test_loader:
        output = model(data[0])
        _, pred = torch.max(output.data, 1)
        pred = pred.tolist()
        for predict in pred:
            name = classes[predict]
            file_name = os.path.basename(files_name[i][0])
            test_y_list.append("{},{}".format(file_name, name))
            i += 1
    test_y_list = sorted(test_y_list, key=lambda x: int(x.split('.')[0]))
    return test_y_list


def write_to_file(predict_y):
    with open('test_y', 'w') as file:
        for y in predict_y:
            file.write("%s\n" % y)
    file.close()


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


if __name__ == "__main__":
    trainLoader, validLoader, testLoader = load_data()
    lr = 0.001
    model = CNNnetwork()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_list = list()
    train_accuracy_list = list()
    valid_loss_list = list()
    valid_accuracy_list = list()

    for epoch in range(30):
        loss, accuracy = train(epoch, model, trainLoader, optimizer, criterion)
        train_loss_list.append(loss)
        train_accuracy_list.append(accuracy)
        valid_loss, valid_accuracy = valid(model, validLoader, criterion)
        valid_loss_list.append(valid_loss)
        valid_accuracy_list.append(valid_accuracy)

    prediction = test(model, testLoader)
    # draw_loss(train_loss_list, valid_loss_list)
    # draw_accuracy(train_accuracy_list, valid_accuracy_list)
    write_to_file(prediction)
