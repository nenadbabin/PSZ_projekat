from data_interpreter.main_data_interpreter import Database
import pandas as pd
from linear_regression.lin_reg import LinearRegression as MyLinearRegression, LinReg
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from utility.helpers import load_data


def main():
    x_values, y_values = load_data()


    # lr_model = LinearRegression()
    # lr_model.fit(x_values.to_numpy(), y_values)
    # lr_preds = lr_model.predict(x_values)
    # print(lr_preds)
    # mse = mean_squared_error(y_values, lr_preds)
    # print(mse)

    # linear_regression = MyLinearRegression(input_data=x_values,
    #                                      correct_output_values=y_values,
    #                                      alpha=0.01)
    # linear_regression.train()

    linear_regression = LinReg()
    linear_regression.fit(x_values.to_numpy(), y_values.to_numpy())
    weights = linear_regression.weights
    print(weights)
    value = linear_regression.predict(x_values.iloc[0])
    print(value)
    pass


if __name__ == "__main__":
    main()