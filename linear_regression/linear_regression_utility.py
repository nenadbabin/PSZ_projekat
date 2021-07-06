import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame


def plot_error_function(loss, epochs):
    plt.plot(epochs, loss)
    plt.xlabel('Iteracija')
    plt.ylabel('Greska')
    plt.title('Vrednost funkcije greske po iteracijama')
    plt.savefig(f"pictures/Vrednost fukncije greske po iteracijama.png")
    plt.show()


def split_data_lr(x_values_data_set: np.ndarray, y_values_data_set: np.ndarray, split_ratio: float = 0.3):

    x_values_data_set = DataFrame(x_values_data_set)
    y_values_data_set = DataFrame(y_values_data_set)
    x_values_data_set['cena'] = y_values_data_set

    number_of_data = x_values_data_set.shape[0]
    test_num = int(number_of_data * split_ratio)
    train_num = number_of_data - test_num

    shuffled_data = x_values_data_set.sample(frac=1).reset_index(drop=True)
    test_data = shuffled_data.head(test_num).copy(deep=True)
    train_data = shuffled_data.tail(train_num).copy(deep=True)

    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)

    y_train = train_data.iloc[:, -1:]
    y_test = test_data.iloc[:, -1:]

    x_train = train_data.iloc[:, :-1]
    x_test = test_data.iloc[:, :-1]

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()

    return x_train, y_train, x_test, y_test
