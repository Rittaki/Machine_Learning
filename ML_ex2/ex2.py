import numpy as np
import sys
import random

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
    print("Rita")

if __name__ == '__main__':
    main()