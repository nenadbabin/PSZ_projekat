from data_interpreter.main_data_interpreter import Database
import pandas as pd


def load_data():
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
                          'lift', 'tip_objekta', 'cena']

    x_values = data_frame[['kvadratura',
                           # 'tip_objekta',
                           'broj_soba',
                           'spratnost']]
    y_values = data_frame[['cena']]

    return x_values, y_values
