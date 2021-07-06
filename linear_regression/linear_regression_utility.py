import numpy as np
import matplotlib.pyplot as plt


def plot_error_function(loss, epochs):
    plt.plot(epochs, loss)
    plt.xlabel('Iteracija')
    plt.ylabel('Greska')
    plt.title('Vrednost funkcije greske po iteracijama')
    plt.savefig(f"pictures/Vrednost fukncije greske po iteracijama.png")
    plt.show()

def split_data_lr(x_values_data_set, y_values_data_set, test_size: float = 0.3):
    np.random.seed(0)
    indices = np.random.permutation(len(x_values_data_set))
    data_test_size = int(x_values_data_set.shape[0] * test_size)

    train_indices = indices[data_test_size:]
    test_indices = indices[:data_test_size]
    x_train = x_values_data_set[train_indices]
    y_train = y_values_data_set[train_indices]
    x_test = x_values_data_set[test_indices]
    y_test = y_values_data_set[test_indices]
    return x_train, y_train, x_test, y_test
