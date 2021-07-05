import numpy as np


def split_data_lr(X, y, test_size=0.25,):
    np.random.seed(0)
    indices = np.random.permutation(len(X))
    data_test_size = int(X.shape[0] * test_size)

    train_indices = indices[data_test_size:]
    test_indices = indices[:data_test_size]
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    return X_train, y_train, X_test, y_test
