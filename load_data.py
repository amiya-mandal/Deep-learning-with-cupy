import pandas as pd 
import cupy as cp 

def load_mnist():
    """
    this is for loading mnist data
    """
    train = pd.read_csv("mnist-in-csv/mnist_train.csv")
    y_train = cp.array(train.iloc[:, 0])
    x_train = cp.array(train.iloc[:, 1:])
    y_train = y_train.reshape((1, 60000))
    train_data = zip(x_train, y_train)

    test = pd.read_csv("mnist-in-csv/mnist_test.csv")
    y_test = cp.array(test.iloc[:, 0])
    x_test = cp.array(test.iloc[:, 1:])
    y_test = y_test.reshape((1, 10000))
    test_data = zip(x_test, y_test)

    return train_data, test_data

