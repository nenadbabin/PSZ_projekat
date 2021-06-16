import numpy as np
from random import random
from typing import List
from pandas import DataFrame, Series
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, input_data: DataFrame, correct_output_values: DataFrame, alpha: float = None):
        if alpha:
            self.alpha: float = alpha
        else:
            self.alpha: float = 1.0
        self.number_of_weights: int = input_data.shape[1]
        # self.number_of_data: int = input_data.shape[0]
        # self.number_of_weights: int = 5
        self.number_of_data: int = 5
        self.weights: List[float] = [0.0 for _ in range(self.number_of_weights)]   # weights w1..wn
        self.bias: float = 0.0   # weight w0
        self.input_data: DataFrame = input_data.head(5)
        self.correct_output_values: DataFrame = correct_output_values.head(5)

    def train(self):
        # Gradient descent
        convergence: bool = False
        iteration: int = 0
        while not convergence:
            old_weights: List[float] = self.weights.copy()   # Copy whole array, not reference!
            temp_weights: List[float] = [0.0 for _ in range(self.number_of_weights)]
            old_bias: float = self.bias
            temp_bias: float = 0.0

            # Calculate new weights
            # calculate w0 (bias)
            temp_bias = old_bias - (self.alpha * (1 / self.number_of_data) * self.__calculate_diff_of_h_and_y_w0())
            for i in range(0, self.number_of_weights):
                # calculate w1...wn
                temp_weights[i] = \
                    old_weights[i] - (self.alpha * (1 / self.number_of_data) * self.__calculate_diff_of_h_and_y_w1_wn(i))

            # Update new weights
            self.bias = temp_bias
            for i in range(0, self.number_of_weights):
                self.weights[i] = temp_weights[i]

            # Check if convergence occurred
            convergence = True
            if abs(self.bias - old_bias) > 0.001:
                convergence = False
            else:
                for i in range(0, self.number_of_weights):
                    if abs(self.weights[i] - old_weights[i]) > 0.001:
                        convergence = False
                        break

            iteration = iteration + 1
            print(f"Iteration: {iteration}")
            print(f"Weights: {self.bias}, {self.weights}")
            print(f"Error: {self.error_function()}")
            if iteration == 10:
                convergence = True

        pass

    def __calculate_diff_of_h_and_y_w0(self) -> float:
        sum_diff_of_all_data: float = 0.0

        for i in range(0, self.number_of_data):
            correct_output_value_data_i, = self.correct_output_values.iloc[i].values.tolist()
            sum_diff_of_all_data += self.__h(self.input_data.iloc[i]) - correct_output_value_data_i

        return sum_diff_of_all_data

    def __calculate_diff_of_h_and_y_w1_wn(self, x_num) -> float:
        sum_diff_of_all_data: float = 0.0

        for i in range(0, self.number_of_data):
            correct_output_value_data_i,  = self.correct_output_values.iloc[i].values.tolist()   # list only has one element
            xn = self.input_data.iloc[i].values.tolist()[x_num]
            sum_diff_of_all_data += (self.__h(self.input_data.iloc[i]) - correct_output_value_data_i) * xn

        return sum_diff_of_all_data

    def __h(self, x: Series) -> float:
        x_values: List[float] = \
            x.values.tolist()   # List of x vales (array_index -> x value | 0 -> x1, 1 -> x2, ..., n1-1 -> xn)
        h_value: float = self.bias
        for i in range(0, self.number_of_weights):   # type: int
            h_value += (self.weights[i] * x_values[i])

        return h_value

    def error_function(self):
        sum_part: float = 0.0
        for i in range(0, self.number_of_data):
            hiphotesis_value: float = self.__h(self.input_data.iloc[i])
            correct_output_value_data_i,  = self.correct_output_values.iloc[i].values.tolist()   # list only has one element
            sum_part += pow(hiphotesis_value - correct_output_value_data_i, 2)

        return (1/(2 * self.number_of_data)) * sum_part

    def predict(self, x: Series) -> float:
        """
        Gets prediction of input data. Model should be trained before using this function.
        :param x: input data x
        :return: y
        """
        return self.__h(x)


# class LinReg:
#     """
#     A class which implements linear regression model with gradient descent.
#     """
#
#     def __init__(self, learning_rate=0.01, n_iterations=10000):
#         self.learning_rate = learning_rate
#         self.n_iterations = n_iterations
#         self.weights, self.bias = None, None
#         self.loss = []
#
#     @staticmethod
#     def _mean_squared_error(y, y_hat):
#         """
#         Private method, used to evaluate loss at each iteration.
#
#         :param: y - array, true values
#         :param: y_hat - array, predicted values
#         :return: float
#         """
#         error = 0
#         for i in range(len(y)):
#             error += (y[i] - y_hat[i]) ** 2
#         return error / len(y)
#
#     def fit(self, X, y):
#         """
#         Used to calculate the coefficient of the linear regression model.
#
#         :param X: array, features
#         :param y: array, true values
#         :return: None
#         """
#         # 1. Initialize weights and bias to zeros
#         self.weights = np.zeros(X.shape[1])
#         self.bias = 0
#
#         # 2. Perform gradient descent
#         for i in range(self.n_iterations):
#             # Line equation
#             y_hat = np.dot(X, self.weights) + self.bias
#             loss = self._mean_squared_error(y, y_hat)
#             self.loss.append(loss)
#
#             # Calculate derivatives
#             partial_w = (1 / X.shape[0]) * (2 * np.dot(X.T, (y_hat - y)))
#             partial_d = (1 / X.shape[0]) * (2 * np.sum(y_hat - y))
#
#             # Update the coefficients
#             self.weights -= self.learning_rate * partial_w
#             self.bias -= self.learning_rate * partial_d


class multipleLinearRegression():

    def __init__(self):
        # No instance Variables required
        pass

    def forward(self, X, y, W):
        """
        Parameters:
        X (array) : Independent Features
        y (array) : Dependent Features/ Target Variable
        W (array) : Weights

        Returns:
        loss (float) : Calculated Sqaured Error Loss for y and y_pred
        y_pred (array) : Predicted Target Variable
        """
        y_pred = sum(W * X)
        loss = ((y_pred - y) ** 2) / 2  # Loss = Squared Error, we introduce 1/2 for ease in the calculation
        return loss, y_pred

    def updateWeights(self, X, y_pred, y_true, W, alpha, index):
        """
        Parameters:
        X (array) : Independent Features
        y_pred (array) : Predicted Target Variable
        y_true (array) : Dependent Features/ Target Variable
        W (array) : Weights
        alpha (float) : learning rate
        index (int) : Index to fetch the corresponding values of W, X and y

        Returns:
        W (array) : Update Values of Weight
        """
        for i in range(X.shape[1]):
            # alpha = learning rate, rest of the RHS is derivative of loss function
            W[i] -= (alpha * (y_pred - y_true[index]) * X[index][i])
        return W

    def train(self, X, y, epochs=10, alpha=0.001, random_state=0):
        """
        Parameters:
        X (array) : Independent Feature
        y (array) : Dependent Features/ Target Variable
        epochs (int) : Number of epochs for training, default value is 10
        alpha (float) : learning rate, default value is 0.001

        Returns:
        y_pred (array) : Predicted Target Variable
        loss (float) : Calculated Sqaured Error Loss for y and y_pred
        """

        num_rows = X.shape[0]  # Number of Rows
        num_cols = X.shape[1]  # Number of Columns
        W = np.random.randn(1, num_cols) / np.sqrt(num_rows)  # Weight Initialization

        # Calculating Loss and Updating Weights
        train_loss = []
        num_epochs = []
        train_indices = [i for i in range(X.shape[0])]
        for j in range(epochs):
            cost = 0
            np.random.seed(random_state)
            np.random.shuffle(train_indices)
            for i in train_indices:
                loss, y_pred = self.forward(X[i], y[i], W[0])
                cost += loss
                W[0] = self.updateWeights(X, y_pred, y, W[0], alpha, i)
            train_loss.append(cost)
            num_epochs.append(j)
        return W[0], train_loss, num_epochs

    def test(self, X_test, y_test, W_trained):
        """
        Parameters:
        X_test (array) : Independent Features from the Test Set
        y_test (array) : Dependent Features/ Target Variable from the Test Set
        W_trained (array) : Trained Weights
        test_indices (list) : Index to fetch the corresponding values of W_trained,
                              X_test and y_test

        Returns:
        test_pred (list) : Predicted Target Variable
        test_loss (list) : Calculated Sqaured Error Loss for y and y_pred
        """
        test_pred = []
        test_loss = []
        test_indices = [i for i in range(X_test.shape[0])]
        for i in test_indices:
            loss, y_test_pred = self.forward(X_test[i], W_trained, y_test[i])
            test_pred.append(y_test_pred)
            test_loss.append(loss)
        return test_pred, test_loss

    def predict(self, W_trained, X_sample):
        prediction = sum(W_trained * X_sample)
        return prediction

    def plotLoss(self, loss, epochs):
        """
        Parameters:
        loss (list) : Calculated Sqaured Error Loss for y and y_pred
        epochs (list): Number of Epochs

        Returns: None
        Plots a graph of Loss vs Epochs
        """
        plt.plot(epochs, loss)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss')
        plt.title('Plot Loss')
        plt.show()
