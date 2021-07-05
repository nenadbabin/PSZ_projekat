import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionPSZ:

    def __init__(self, x_values: np.ndarray, true_y_values: np.ndarray, alpha: float = 0.0001, num_of_iter: int = 200):
        self.x_values = x_values
        self.true_y_values = true_y_values
        self.alpha = alpha
        self.num_of_iter = num_of_iter

        self.num_of_data = self.x_values.shape[0]
        self.num_of_features = self.x_values.shape[1]
        self.weights = (np.random.randn(1, self.num_of_features) / np.sqrt(self.num_of_data))[0]
        self.bias = (np.random.randn(1, 1) / np.sqrt(self.num_of_data))[0][0]

        pass

    def __h(self, single_data_x_values: np.ndarray) -> float:
        y_predicted = self.bias + sum(self.weights * single_data_x_values)
        return y_predicted

    def predict(self, single_data_x_values: np.ndarray) -> float:
        return self.__h(single_data_x_values)

    @staticmethod
    def __loss_function(y_predicted: float, true_y_value: float) -> float:
        loss = pow(y_predicted - true_y_value, 2) / 2
        return loss

    def __update_weights(self, single_data_y_pred: float, index: int):
        self.bias -= (self.alpha * (single_data_y_pred - self.true_y_values[index]))
        for i in range(self.num_of_features):
            self.weights[i] -= (self.alpha * (single_data_y_pred - self.true_y_values[index]) * self.x_values[index][i])

    def train(self):
        train_loss = []
        num_epochs = []
        train_indices = [i for i in range(self.num_of_data)]
        for j in range(self.num_of_iter):
            cost = 0
            np.random.seed(0)
            np.random.shuffle(train_indices)
            for i in train_indices:
                y_predicted = self.__h(self.x_values[i])
                loss = self.__loss_function(y_predicted, self.true_y_values[i])
                cost += loss
                self.__update_weights(y_predicted, i)
            train_loss.append(cost)
            num_epochs.append(j)
        return train_loss, num_epochs

    def test(self, x_test_set: np.ndarray, y_test_set: np.ndarray):
        test_pred = []
        test_loss = []
        test_indices = [i for i in range(x_test_set.shape[0])]
        for i in test_indices:
            y_test_pred = self.__h(x_test_set[i])
            loss = self.__loss_function(y_test_pred, y_test_set[i])
            test_pred.append(y_test_pred)
            test_loss.append(loss)
        return test_pred, test_loss

    @staticmethod
    def plot_loss(loss, epochs):
        plt.plot(epochs, loss)
        plt.xlabel('Iteracija')
        plt.ylabel('Greska')
        plt.title('Vrednost fukncije greske po teracijama')
        plt.show()
