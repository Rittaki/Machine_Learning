import numpy as np
import sys
import random

def perceptron(train_x, train_y):
    eta = 0.1 # change later
    epochs = 20 # change later
    weights = np.zeros((3, train_x.shape[1]))
    for t in range(epochs):
        for x_i, y_i in zip(train_x, train_y):
            y_hat = np.argmax(np.dot(weights, x_i))
            if y_hat != int(y_i):
                weights[int(y_i), :] = weights[int(y_i), :] + eta * x_i
                weights[y_hat, :] = weights[y_hat, :] - eta * x_i
    return weights


def minmax(examples):
    columns = examples.shape[1]
    for i in range(columns):
        x = examples[:, i]
        examples[:, i] = (x - x.min()) / (x.max() - x.min())
        # or try:
        # examples[:,i] = (examples[:,i] - examples[:,i].min()) / (examples[:,i].max() - examples[:,i].min())

def z_score(examples):
    columns = examples.shape[1]
    for i in range(columns):
        mean = np.mean(examples[:, i])
        deviation = np.std(examples[:, i])
        if deviation != 0:
            examples[:, i] = (examples[:, i] - mean) / deviation

def main():
    train_x_path, train_y_path, test_x_path, out_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    out = open(out_file, "a")
    test_x = np.loadtxt(test_x_path, delimiter=',')
    train_x = np.loadtxt(train_x_path, delimiter=',')
    train_y = np.genfromtxt(train_y_path)
    # minmax(train_x)
    z_score(train_x)
    perceptron_weights = perceptron(train_x, train_y)
    print("Rita")

if __name__ == '__main__':
    main()