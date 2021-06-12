from random import random
from typing import List

from pandas import DataFrame, Series


class LinearRegression:
    def __init__(self, input_data: DataFrame, correct_output_values: DataFrame, alpha: float = None):
        if alpha:
            self.alpha: float = alpha
        else:
            self.alpha: float = 1
        # self.number_of_weights: int = input_data.shape[1] + 1
        # self.number_of_data: int = input_data.shape[0]
        self.number_of_weights: int = 4
        self.number_of_data: int = 10
        self.weights: List[float] = [random() for _ in range(self.number_of_weights)]
        self.input_data: DataFrame = input_data.head(10)
        self.correct_output_values: DataFrame = correct_output_values.head(10)

    def train(self):
        # Gradient descent
        convergence: bool = False
        while not convergence:
            old_weights: List[float] = self.weights.copy()   # Copy whole array, not reference!
            temp_weights: List[float] = [0.0 for _ in range(self.number_of_weights)]

            # Calculate new weights
            for i in range(0, self.number_of_weights):
                if i == 0:   # calculate w0
                    temp_weights[i] = \
                        old_weights[i] - self.alpha * (1/self.number_of_data) * self.calculate_diff_of_h_and_y_w0()
                else:   # calculate w1...wn
                    temp_weights[i] = \
                        old_weights[i] - self.alpha * (1 / self.number_of_data) * self.calculate_diff_of_h_and_y_w1_wn(i)

            # Update new weights
            for i in range(0, self.number_of_weights):
                self.weights[i] = temp_weights[i]

            # Check if convergence occurred
            convergence = True
            for i in range(0, self.number_of_weights):
                if abs(self.weights[i] - old_weights[i]) > 0.001:
                    convergence = False
                    break

    def calculate_diff_of_h_and_y_w0(self) -> float:
        sum_diff_of_all_data = 0

        for i in range(0, self.number_of_data):
            sum_diff_of_all_data += \
                self.h(self.input_data.iloc[i]) - self.correct_output_values.iloc[i].values.tolist()[0]

        return sum_diff_of_all_data

    def calculate_diff_of_h_and_y_w1_wn(self, x_num) -> float:
        sum_diff_of_all_data: int = 0

        for i in range(0, self.number_of_data):
            sum_diff_of_all_data += (self.h(self.input_data.iloc[i]) - self.correct_output_values.iloc[i].values.tolist()[0]) * self.input_data.iloc[i].values.tolist()[x_num - 1]

        return sum_diff_of_all_data

    def h(self, x: Series) -> float:
        x_values: List[float] = \
            x.values.tolist()   # List of x vales (array_index -> x value | 0 -> x1, 1 -> x2, ..., n1-1 -> xn)
        h_value: float = 0
        for i in range(0, self.number_of_weights):   # type: int
            if i == 0:
                h_value += self.weights[i]
            else:
                h_value += self.weights[i] * x_values[i-1]

        return h_value
