import numpy as np


class LinearRegressionPSZ:

    def __init__(self, x_values: np.ndarray, true_y_values: np.ndarray, alpha: float = 0.0001, num_of_iter: int = 200):
        self.__x_values = x_values
        self.__true_y_values = true_y_values
        self.__alpha = alpha
        self.__num_of_iter = num_of_iter

        self.__num_of_data = self.__x_values.shape[0]
        self.__num_of_features = self.__x_values.shape[1]
        self.__weights = (np.random.randn(1, self.__num_of_features) / np.sqrt(self.__num_of_data))[0]
        self.__bias = (np.random.randn(1, 1) / np.sqrt(self.__num_of_data))[0]

    def num_of_features(self) -> int:
        return self.__num_of_features

    def __h(self, single_data_x_values: np.ndarray) -> float:
        y_predicted = self.__bias + sum(self.__weights * single_data_x_values)
        return y_predicted

    def predict(self, single_data_x_values: np.ndarray) -> float:
        return self.__h(single_data_x_values)

    @staticmethod
    def __square_diff(y_predicted: float, true_y_value: float) -> float:
        loss = pow(y_predicted - true_y_value, 2)
        return loss

    def __update_weights(self, single_data_y_pred: float, index: int):
        self.__bias -= (self.__alpha * (single_data_y_pred - self.__true_y_values[index]))
        for i in range(self.__num_of_features):
            self.__weights[i] -= (self.__alpha * (single_data_y_pred - self.__true_y_values[index]) * self.__x_values[index][i])

    def train(self):
        train_error = []
        num_epochs = []
        train_indices = [i for i in range(self.__num_of_data)]
        for j in range(self.__num_of_iter):
            square_diff_sum = 0
            np.random.seed(0)
            np.random.shuffle(train_indices)
            old_bias = self.__bias.copy()
            old_weights = self.__weights.copy()
            for i in train_indices:
                y_predicted = self.__h(self.__x_values[i])
                square_diff = self.__square_diff(y_predicted, self.__true_y_values[i])
                square_diff_sum += square_diff
                self.__update_weights(y_predicted, i)
            train_error.append(square_diff_sum / (2 * self.__num_of_data))
            num_epochs.append(j)

            converged = True
            for i in range(0, len(old_weights)):
                if old_weights[i] != self.__weights[i]:
                    converged = False
                    break

            if converged and old_bias == self.__bias:
                break

        return train_error, num_epochs

    def test(self, x_test_set: np.ndarray, y_test_set: np.ndarray):
        test_pred = []
        square_diffs = []
        test_indices = [i for i in range(x_test_set.shape[0])]
        for i in test_indices:
            y_test_pred = self.__h(x_test_set[i])
            square_diff = self.__square_diff(y_test_pred, y_test_set[i])
            test_pred.append(y_test_pred)
            square_diffs.append(square_diff)

        RMSE = np.sqrt((np.sum(square_diffs)) / x_test_set.shape[0])
        return test_pred, RMSE
