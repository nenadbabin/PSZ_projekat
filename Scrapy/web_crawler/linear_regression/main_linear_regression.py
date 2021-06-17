import numpy as np
from linear_regression.lin_reg import MultipleLinearRegression
from linear_regression.linear_regression_utility import split_data
from utility.helpers import load_data


def main():
    x_values, y_values = load_data()

    x_values = x_values.to_numpy()
    y_values = y_values.to_numpy()

    for i in range(x_values.shape[1]):
        x_values[:, i] = (x_values[:, i] - int(np.mean(x_values[:, i]))) / np.std(x_values[:, i])

    x_train, y_train, x_test, y_test = split_data(x_values, y_values)

    regression = MultipleLinearRegression()
    weights_trained, train_loss, num_epochs = regression.train(x_train, y_train, epochs=200, alpha=0.00001)
    test_pred, test_loss = regression.test(x_test, y_test, weights_trained)
    regression.plotLoss(train_loss, num_epochs)


if __name__ == "__main__":
    main()
