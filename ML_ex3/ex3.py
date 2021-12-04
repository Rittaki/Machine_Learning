import sys
import numpy as np
# from scipy.special import softmax
import matplotlib.pyplot as plt

# Example backpropagation code for binary classification with 2-layer
# neural network (single hidden layer)

# sigmoid = lambda x: 1 / (1 + np.exp(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

def loss_func(h2, y):
    vec = np.zeros((10, 1))
    vec[int(y)] = 1
    res = np.sum(-vec * np.log(h2))
    return res

def fprop(W1, b1, W2, b2, x):
    # Follows procedure given in notes
    z1 = np.dot(W1, x) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(W2, h1) + b2
    h2 = softmax(z2)
    # loss = loss_func(h2, y)
    return z1, h1, z2, h2

def bprop(Z1, h1, Z2, h2, W1, W2, x, y):
    # Follows procedure given in notes
    m = y.size
    one_hot_y = one_hot(y)
    dz2 = (h2 - one_hot_y)                                #  dL/dz2
    dW2 = 1 / m * np.dot(dz2, h1.T)                       #  dL/dz2 * dz2/dw2
    db2 = 1 / m * np.sum(dz2)                                     #  dL/dz2 * dz2/db2
    dz1 = np.dot(W2.T, dz2) * sigmoid(Z1) * (1-sigmoid(Z1))   #  dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = 1 / m * np.dot(dz1, x.T)                        #  dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = 1 / m * np.sum(dz1)                                     #  dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def one_hot(Y):
    one_hot_y = np.zeros((Y.size, 10))
    one_hot_y[np.arange(Y.size), Y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

def parameters():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return W1, b1, W2, b2

def get_predictions(h2):
    return np.argmax(h2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

if __name__ == '__main__':
    train_x = np.loadtxt(sys.argv[1])
    train_y = np.loadtxt(sys.argv[2])
    test_x = np.loadtxt(sys.argv[3])
    # print(train_x.shape)
    train_y = train_y.reshape(train_y.shape[0], 1)
    # print(train_y.shape)
    train_data = np.append(train_y, train_x, axis=1)
    # print(train_data.shape)
    m, n = train_data.shape
    np.random.shuffle(train_data)
    data = train_data[0:m].T
    train_y = data[0]
    train_y = train_y.reshape(1, -1).astype(int)
    train_x = data[1:n]

    norm = 0.99 / 255
    norm_train_x = train_x * norm + 0.01
    # norm_test_x = test_x * norm + 0.01

    W1, b1, W2, b2 = parameters()
    for i in range(500):
        Z1, h1, Z2, h2  = fprop(W1, b1, W2, b2, norm_train_x)
        dW1, db1, dW2, db2 = bprop(Z1, h1, Z2, h2, W1, W2, norm_train_x, train_y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, 0.1)
        if i % 5 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(h2)
            print(get_accuracy(predictions, train_y))

    # image_index = 50000 # You may select anything up to 50000
    # print(train_y[image_index])
    # plt.imshow(train_x[image_index].reshape(28, -1), cmap='gray')
    # plt.show()
