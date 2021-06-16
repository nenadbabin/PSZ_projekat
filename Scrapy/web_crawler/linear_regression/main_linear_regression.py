from sklearn.model_selection import train_test_split

from data_interpreter.main_data_interpreter import Database
import pandas as pd
import numpy as np
from linear_regression.lin_reg import LinearRegression as MyLinearRegression, multipleLinearRegression
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

    # data = pd.read_csv('train.csv')
    # x_values = data[['x']]
    # y_values = data[['y']]

    # linear_regression = MyLinearRegression(input_data=x_values,
    #                                      correct_output_values=y_values,
    #                                      alpha=0.01)
    # linear_regression.train()

    x_values = x_values.to_numpy()
    y_values = y_values.to_numpy()

    for i in range(x_values.shape[1] - 2):
        x_values[:, i] = (x_values[:, i] - int(np.mean(x_values[:, i]))) / np.std(x_values[:, i])

    x_train, y_train, x_test, y_test = split_data(x_values, y_values)

    # x_train, x_test, y_train, y_test = \
    #     train_test_split(x_values, y_values, test_size=0.25, random_state=1121218, stratify=y_values)

    regressor = multipleLinearRegression()
    W_trained, train_loss, num_epochs = regressor.train(x_train, y_train, epochs=200, alpha=0.00001)
    test_pred, test_loss = regressor.test(x_test, y_test, W_trained)
    regressor.plotLoss(train_loss, num_epochs)

    # linear_regression = LinReg()
    # linear_regression.fit(x_values.to_numpy(), y_values.to_numpy())
    # weights = linear_regression.weights
    # print(weights)
    # value = linear_regression.predict(x_values.iloc[0])
    # print(value)
    pass


# The method "split_data" splits the given dataset into trainset and testset
# This is similar to the method "train_test_split" from "sklearn.model_selection"
def split_data(X,y,test_size=0.2,random_state=0):
    np.random.seed(random_state)                  #set the seed for reproducible results
    indices = np.random.permutation(len(X))       #shuffling the indices
    data_test_size = int(X.shape[0] * test_size)  #Get the test size


if __name__ == "__main__":
    main()