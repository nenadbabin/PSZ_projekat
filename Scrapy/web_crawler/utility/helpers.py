import math
from typing import List

from pandas import DataFrame

from data_interpreter.main_data_interpreter import Database
import pandas as pd

X_FEATURE_LIST = ['kvadratura',
                  'broj_soba',
                  'spratnost',
                  'udaljenost_od_centra',
                  'tip_objekta_klasa']

Y_FEATURE_LIST = ['cena']

CENTER_OF_BELGRADE_X = 44.8237
CENTER_OF_BELGRADE_Y = 20.4526


def load_data(x_features: List[str] = X_FEATURE_LIST):
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
