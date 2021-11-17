import numpy as np
import sys
import random

# def knn(train_x, train_y, test_x):
#     epochs = 50 # change later
#     k = 5 # change later
#     weights = np.zeros((3, train_x.shape[1]))
#     for t in range(epochs):
#         for x_test in test_x:
#
#
#     return weights

def passive_aggressive(train_x, train_y):
    epochs = 50 # change later
    weights = np.zeros((3, train_x.shape[1]))
    for t in range(epochs):
        # shuffled = list(zip(train_x, train_y))
        # np.random.shuffle(shuffled)
        # x, y = zip(*shuffled)
        for x_i, y_i in zip(train_x, train_y):
            y_hat = np.argmax(np.dot(weights, x_i))
            if y_hat != int(y_i):
                loss = max(0, 1 - np.dot(weights[int(y_i)], x_i) + np.dot(weights[y_hat], x_i))
                norm = 2 * ((np.linalg.norm(x_i)) ** 2)
                if norm != 0:
                    tau = loss/norm
                    weights[int(y_i), :] = weights[int(y_i), :] + tau * x_i
                    weights[y_hat, :] = weights[y_hat, :] - tau * x_i
    return weights

def svm(train_x, train_y):
    eta = 0.01 # change later
    lamda = 0.1
    epochs = 50
    weights = np.zeros((3, train_x.shape[1]))
    for t in range(epochs):
        for x_i, y_i in zip(train_x, train_y):
            y_hat = np.argmax(np.dot(weights, x_i))
            if y_hat != int(y_i):
                weights[int(y_i), :] = (1 - eta * lamda) * weights[int(y_i), :] + eta * x_i
                weights[y_hat, :] = (1 - eta * lamda) * weights[y_hat, :] - eta * x_i
            for i in range(weights.shape[0]):
                if i != int(y_i) and i != y_hat:
                    weights[i, :] = (1 - eta * lamda) * weights[i, :]
    return weights

def perceptron(train_x, train_y):
    eta = 0.1 # change later
    epochs = 50 # change later
    weights = np.zeros((3, train_x.shape[1]))
    for t in range(epochs):
        # shuffled = list(zip(train_x, train_y))
        # np.random.shuffle(shuffled)
        # x, y = zip(*shuffled)
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

def test(test_examples, pa_weights, per_weights, svm_weights, out):
    # number of tests
    m = test_examples.shape[0]
    z_score(test_examples)
    for i in range(m):
        perceptron_yhat = np.argmax(np.dot(per_weights, test_examples[i]))
        pa_yhat = np.argmax(np.dot(pa_weights, test_examples[i]))
        svm_yhat = np.argmax(np.dot(svm_weights, test_examples[i]))
        # out.write(f"knn: {knn_yhat}, perceptron: {perceptron_yhat}, svm: {svm_yhat}, pa: {pa_yhat}\n")
        out.write(f"perceptron: {perceptron_yhat}, svm: {svm_yhat}, pa: {pa_yhat}\n")

def main():
    train_x_path, train_y_path, test_x_path, out_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    out = open(out_file, "a")
    test_x = np.loadtxt(test_x_path, delimiter=',')
    train_x = np.loadtxt(train_x_path, delimiter=',')
    train_y = np.genfromtxt(train_y_path)
    # minmax(train_x)
    z_score(train_x)
    z_score(test_x)
    perceptron_weights = perceptron(train_x, train_y)
    pa_weights = passive_aggressive(train_x, train_y)
    svm_weights = svm(train_x, train_y)
    # knn_weights = knn(train_x, train_y, test_x)
    test(test_x, pa_weights, perceptron_weights, svm_weights, out)
    print("Rita")

if __name__ == '__main__':
    main()