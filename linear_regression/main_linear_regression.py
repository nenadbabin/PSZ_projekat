import numpy as np
from linear_regression.lin_reg import LinearRegressionPSZ
from linear_regression.linear_regression_utility import split_data_lr
from utility.helpers import load_data, load_data_from_csv, plot


def main():
    x_values, y_values = load_data_from_csv(path="../data/filtered_data.csv")

    plot(x_values['kvadratura'], y_values['cena'], "Raspodela kvadratura-cena", "kvadratura", "cena")
    plot(x_values['broj_soba'], y_values['cena'], "Raspodela broj_soba-cena", "broj_soba", "cena")
    plot(x_values['spratnost'], y_values['cena'], "Raspodela spratnost-cena", "spratnost", "cena")
    plot(x_values['udaljenost_od_centra'], y_values['cena'], "Raspodela udaljenost_od_centra-cena", "udaljenost_od_centra", "cena")
    plot(x_values['tip_objekta_klasa'], y_values['cena'], "Raspodela tip_objekta_klasa-cena", "tip_objekta_klasa", "cena")

    x_values = x_values.to_numpy()
    y_values = y_values.to_numpy()

    for i in range(x_values.shape[1]):
        x_values[:, i] = (x_values[:, i] - int(np.mean(x_values[:, i]))) / np.std(x_values[:, i])

    y_values = (y_values - int(np.mean(y_values))) / np.std(y_values)

    x_train, y_train, x_test, y_test = split_data_lr(x_values, y_values)

    regression = LinearRegressionPSZ(x_train, y_train, alpha=0.00001, num_of_iter=500)
    train_cost, num_epochs = regression.train()
    _, RMSE_train = regression.test(x_train, y_train)
    print(f"RMSE (train): {RMSE_train}")
    _, RMSE_test = regression.test(x_test, y_test)
    print(f"RMSE (test): {RMSE_test}")
    regression.plot_loss(train_cost, num_epochs)


if __name__ == "__main__":
    main()
