import math
from typing import List
from pandas import DataFrame
from data_interpreter.main_data_interpreter import Database
import pandas as pd
import matplotlib.pyplot as plt

X_FEATURE_LIST = ['kvadratura',
                  'broj_soba',
                  'spratnost',
                  'udaljenost_od_centra',
                  'tip_objekta_klasa']

Y_FEATURE_LIST = ['cena']

CENTER_OF_BELGRADE_X = 44.8237
CENTER_OF_BELGRADE_Y = 20.4526


def load_data(x_features: List[str] = X_FEATURE_LIST, path: str = None):

    if path is None:
        database = Database(host="localhost",
                            user="root",
                            password="nenad",
                            database="psz_projekat")

        query = "select * from psz_projekat.nekretnina " \
                "where grad = 'Beograd' and tip_ponude = 'Prodaja' and tip_nekretnine = 'Stan';"

        database.cursor.execute(query)
        data_frame = pd.DataFrame(database.cursor.fetchall())
        data_frame.columns = ['id', 'tip_ponude', 'tip_nekretnine',
                              'broj_soba', 'spratnost', 'sprat',
                              'povrsina_placa', 'grejanje', 'grad',
                              'lokacija', 'mikrolokacija', 'kvadratura',
                              'parking', 'uknjizenost', 'terasa',
                              'lift', 'tip_objekta', 'cena', 'x_pos', 'y_pos']
    else:
        data_frame = pd.read_csv(path)

    data_frame['udaljenost_od_centra'] = 0.0
    data_frame['tip_objekta_klasa'] = 0.0

    for i in range(0, data_frame.shape[0]):
        data_frame['udaljenost_od_centra'][i] = calculate_distance(CENTER_OF_BELGRADE_X, CENTER_OF_BELGRADE_Y,
                                                                   float(data_frame['x_pos'][i]), float(data_frame['y_pos'][i]))

        if data_frame['tip_objekta'][i] == 'Novogradnja':
            data_frame['tip_objekta_klasa'][i] = 1
        elif data_frame['tip_objekta'][i] == 'Stara gradnja':
            data_frame['tip_objekta_klasa'][i] = 2
        else:
            data_frame['tip_objekta_klasa'][i] = 3

    # data_frame.to_csv("raw_data.csv", encoding='utf-8', index=True)

    data_frame = data_frame.drop([1462, 4865, 2321, 8269, 8248, 3068, 8750, 5257, 5188, 6549, 4661, 9544,
                                  1451, 1221, 1612, 6332, 1812, 1566, 8419, 1594, 5938, 6911, 6081, 4378,
                                  4440, 4844, 1365, 5994, 1462, 2321, 8750, 5257, 7683, 6647, 7606, 7600,
                                  6292, 7601, 3375, 2686, 1212, 8677, 8342, 6727, 6732, 4815, 1895, 1462,
                                  1462, 2321, 8750, 8749, 2795, 1594, 5938, 5257, 1612, 6322, 1566, 1812,
                                  8419, 8545, 7583, 7311])

    data_frame = data_frame[data_frame.tip_objekta_klasa != 3]

    # data_frame.to_csv("filtered_data.csv", encoding='utf-8', index=True)

    data_frame.reset_index(inplace=True)

    x_values = data_frame[x_features]
    y_values = data_frame[Y_FEATURE_LIST]

    return x_values, y_values


def load_data_from_csv(x_features: List[str] = X_FEATURE_LIST, path: str = None):
    data_frame = pd.read_csv(path)
    x_values = data_frame[x_features]
    y_values = data_frame[Y_FEATURE_LIST]

    return x_values, y_values


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))


def load_all_data() -> DataFrame:
    database = Database(host="localhost",
                        user="root",
                        password="nenad",
                        database="psz_projekat")

    query = "select * from psz_projekat.nekretnina"

    database.cursor.execute(query)
    data_frame = pd.DataFrame(database.cursor.fetchall())
    data_frame.columns = ['id', 'tip_ponude', 'tip_nekretnine',
                          'broj_soba', 'spratnost', 'sprat',
                          'povrsina_placa', 'grejanje', 'grad',
                          'lokacija', 'mikrolokacija', 'kvadratura',
                          'parking', 'uknjizenost', 'terasa',
                          'lift', 'tip_objekta', 'cena', 'x_pos', 'y_pos']

    return data_frame


def plot(x_values, y_values, title, x_axis_label, y_axis_label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(x_values, y_values, label="s", color="blue", marker="o", s=30)
    # for i, xy in enumerate(zip(x_values, y_values)):
    #     ax.annotate(f"{i}", xy=xy, textcoords='data')

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)

    # function to show the plot
    plt.show()
