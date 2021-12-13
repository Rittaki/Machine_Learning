import sys
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(Z):
    Z = Z - np.max(Z)
    return np.exp(Z) / sum(np.exp(Z))

def loss_func(h2, y):
    result = np.sum(-(y * np.log(h2) + (1 - y) * np.log(1 - h2)))
    return result

def fprop(params, x, y):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    # Follows procedure given in notes
    z1 = np.dot(W1, x) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(W2, h1) + b2
    h2 = softmax(z2)
    loss = loss_func(h2.copy(), y.copy())
    ret = {'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
    return ret

def bprop(fprop, W1, W2, x, y):
    # Follows procedure given in notes
    z1, h1, z2, h2 = [fprop[key] for key in ('z1', 'h1', 'z2', 'h2')]
    one_hot_y = one_hot(y)
    dz2 = (h2 - one_hot_y)  # dL/dz2
    dW2 = np.dot(dz2, h1.T)  # dL/dz2 * dz2/dw2
    db2 = dz2  # dL/dz2 * dz2/db2
    sig_z1 = sigmoid(z1)
    dz1 = np.dot(W2.T, dz2) * sig_z1 * (1 - sig_z1)  # dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = np.dot(dz1, x.T)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return dW1, db1, dW2, db2

def update_params(params, dW1, db1, dW2, db2, alpha):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return params

def one_hot(Y):
    one_hot_y = np.zeros((Y.size, 10))
    one_hot_y[np.arange(Y.size), Y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

def parameters():
    W1 = np.random.rand(100, 784) - 0.5
    b1 = np.zeros([100, 1], dtype=float)
    W2 = np.random.rand(10, 100) - 0.5
    b2 = np.zeros([10, 1], dtype=float)
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return params

def get_predictions(h2):
    return np.argmax(h2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def test(test_x, W1, b1, W2, b2):
    open_test_y = open("test_y", "a")
    for x in test_x:
        z1 = np.dot(W1, np.array([x]).T) + b1
        h1 = sigmoid(z1)
        z2 = np.dot(W2, h1) + b2
        h2 = softmax(z2)
        y_predictions = str(np.argmax(h2))
        open_test_y.write(f"{y_predictions}\n")
    open_test_y.close()

def split(train_x, train_y):
    n = int(train_x.shape[1] * 0.2)
    x_valid = train_x[:, 0:n]
    x_data = train_x[:, n:]
    y_valid = train_y.T[0:n]
    y_data = train_y.T[n:]
    return x_data, y_data, x_valid, y_valid

if __name__ == '__main__':
    train_x = np.loadtxt(sys.argv[1])
    train_y = np.loadtxt(sys.argv[2])
    test_x = np.loadtxt(sys.argv[3])
    train_y = train_y.reshape(train_y.shape[0], 1)
    train_data = np.append(train_y, train_x, axis=1)
    m, n = train_data.shape
    np.random.shuffle(train_data)
    data = train_data[0:m].T
    train_y = data[0]
    train_y = train_y.reshape(1, -1).astype(int)
    train_x = data[1:n]
    test_x = test_x

    eta = 0.01
    norm = 0.99 / 255
    norm_train_x = train_x * norm + 0.01
    norm_test_x = test_x * norm + 0.01

    params = parameters()
    best_w1 = params['W1']
    best_b1 = params['b1']
    best_w2 = params['W2']
    best_b2 = params['b2']
    best = 0

    x_data, y_data, x_val, y_val = split(norm_train_x, train_y)

    for i in range(20):
        h2_all = []
        for x, y in zip(x_data.T, y_data):
            x = np.ndarray(shape=(784, 1), buffer=x)
            fprop_cache = fprop(params, x, y)
            dW1, db1, dW2, db2 = bprop(fprop_cache, params['W1'], params['W2'], x, y)
            params = update_params(params, dW1, db1, dW2, db2, eta)
            h2_all.append(fprop_cache['h2'])
        h2_all = np.asarray(h2_all)
        h2_all_new = h2_all.reshape(h2_all.shape[0], (h2_all.shape[1]*h2_all.shape[2]))
        print("Iteration: ", i)
        predictions = get_predictions(h2_all_new.T)
        print(get_accuracy(predictions, y_data.T))

        correct = 0
        for x, y in zip(x_val.T, y_val):
            x = np.ndarray(shape=(784, 1), buffer=x)
            fprop_validation = fprop(params, x, y)
            if np.argmax(fprop_validation['h2']) == y:
                correct += 1
        if (correct / y_val.shape[0]) > best:
            best = correct / y_val.shape[0]
            best_w1 = params['W1'].copy()
            best_w2 = params['W2'].copy()
            best_b1 = params['b1'].copy()
            best_b2 = params['b2'].copy()

    test(norm_test_x, best_w1, best_b1, best_w2, best_b2)
