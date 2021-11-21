import numpy as np
import sys
import random

def knn(train_x, train_y, x_test):
    k = 7 # change later
    classes = [0, 0, 0]
    distances = []
    for x_i in train_x:
        distances.append(np.linalg.norm(x_test - x_i))
    k_distances = sorted(distances)[: k]
    k_neighbours = []
    for dist in k_distances:
        k_neighbours.append(train_x[distances.index(dist)])
    for neighbour in k_neighbours:
        index = np.where(np.all(neighbour == train_x, axis=1))[0][0]
        classes[int(train_y[index])] += 1
    return np.argmax(classes)

def passive_aggressive(train_x, train_y):
    epochs = 50 # change later
    bias = np.ones((len(train_x), 1))
    train_newx = np.append(train_x, bias, axis=1)
    weights = np.zeros((3, train_newx.shape[1]))
    for t in range(epochs):
        for x_i, y_i in zip(train_newx, train_y):
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
    lamda = 0.01
    epochs = 50
    bias = np.ones((len(train_x), 1))
    train_newx = np.append(train_x, bias, axis=1)
    weights = np.zeros((3, train_newx.shape[1]))
    for t in range(epochs):
        for x_i, y_i in zip(train_newx, train_y):
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
    bias = np.ones((len(train_x), 1))
    train_newx = np.append(train_x, bias, axis=1)
    weights = np.zeros((3, train_newx.shape[1]))
    for t in range(epochs):
        for x_i, y_i in zip(train_newx, train_y):
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

def test(test_examples, pa_weights, per_weights, svm_weights, out, train_x, train_y, test_for_knn):
    # number of tests
    bias = np.ones((len(test_examples), 1))
    test_examples_new = np.append(test_examples, bias, axis=1)
    m = test_examples.shape[0]
    z_score(test_examples_new)
    for i in range(m):
        knn_yhat = knn(train_x, train_y, test_for_knn[i])
        perceptron_yhat = np.argmax(np.dot(per_weights, test_examples_new[i]))
        pa_yhat = np.argmax(np.dot(pa_weights, test_examples_new[i]))
        svm_yhat = np.argmax(np.dot(svm_weights, test_examples_new[i]))
        out.write(f"knn: {knn_yhat}, perceptron: {perceptron_yhat}, svm: {svm_yhat}, pa: {pa_yhat}\n")

def main():
    train_x_path, train_y_path, test_x_path, out_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    out = open(out_file, "a")
    test_x = np.loadtxt(test_x_path, delimiter=',')
    train_x = np.loadtxt(train_x_path, delimiter=',')
    train_y = np.genfromtxt(train_y_path)
    train_x_for_knn = train_x.copy()
    train_y_for_knn = train_y.copy()
    test_x_for_knn = test_x.copy()
    # minmax(train_x)
    # minmax(test_x)
    z_score(train_x)
    z_score(test_x)
    perceptron_weights = perceptron(train_x, train_y)
    pa_weights = passive_aggressive(train_x, train_y)
    svm_weights = svm(train_x, train_y)
    test(test_x, pa_weights, perceptron_weights, svm_weights, out, train_x_for_knn, train_y_for_knn, test_x_for_knn)

if __name__ == '__main__':
    main()