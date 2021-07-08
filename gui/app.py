import threading
from typing import List

import pandas as pd
import numpy as np
import wx
import os
import math

from k_nearest_neighbors.KNN import KNN, DistanceMetric
from k_nearest_neighbors.knn_utility import split_data_knn, get_predicted_class, calculate_accuracy
from linear_regression.lin_reg import LinearRegressionPSZ
from linear_regression.linear_regression_utility import split_data_lr, plot_error_function
from utility.helpers import load_data, X_FEATURE_LIST, load_data_from_csv, CENTER_OF_BELGRADE_X, CENTER_OF_BELGRADE_Y


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


class MyFrame(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title='PSZ', size=(800, 400))
        panel = wx.Panel(self)

        # ODLIKE ###############################

        self.static_text_kvadratura = wx.StaticText(panel, label="Kvadratura", pos=(250, 20))
        self.check_box_kvadratura = wx.CheckBox(panel, pos=(400, 20))

        self.static_text_broj_soba = wx.StaticText(panel, label="Broj soba", pos=(250, 40))
        self.check_box_broj_soba = wx.CheckBox(panel, pos=(400, 40))

        self.static_text_spratnost = wx.StaticText(panel, label="Spratnost", pos=(250, 60))
        self.check_box_spratnost = wx.CheckBox(panel, pos=(400, 60))

        self.static_text_udaljenost_od_centra = wx.StaticText(panel, label="Udaljenost od centra", pos=(250, 80))
        self.check_box_udaljenost_od_centra = wx.CheckBox(panel, pos=(400, 80))

        self.static_text_tip_objekta = wx.StaticText(panel, label="Tip objekta", pos=(250, 100))
        self.check_box_tip_objekta = wx.CheckBox(panel, pos=(400, 100))

        # LINEARNA REGRESIJA ###############################

        self.static_text_lr_iteracije = wx.StaticText(panel, label="LINEARNA REGRESIJA", pos=(5, 10))

        self.static_text_lr_iteracije = wx.StaticText(panel, label="Broj iteracija:", pos=(5, 40))
        self.text_ctrl_lr_iteracije = wx.TextCtrl(panel, pos=(80, 40))
        self.text_ctrl_lr_iteracije.SetValue("500")

        self.static_text_lr_alfa = wx.StaticText(panel, label="Alfa:", pos=(5, 70))
        self.text_ctrl_lr_alfa = wx.TextCtrl(panel, pos=(80, 70))
        self.text_ctrl_lr_alfa.SetValue("0.00001")

        self.static_text_lr_test = wx.StaticText(panel, label="Test:", pos=(5, 100))
        self.text_ctrl_lr_test = wx.TextCtrl(panel, pos=(80, 100))
        self.text_ctrl_lr_test.SetValue("0.3")

        self.button_lr_treniraj = wx.Button(panel, label='Linearna regresija', pos=(5, 130))
        self.button_lr_treniraj.Bind(wx.EVT_BUTTON, self.lin_reg_event)

        # KNN ###############################

        self.static_text_knn = wx.StaticText(panel, label="K NAJBLIZIH SUSEDA junk", pos=(5, 170))

        self.static_text_knn_k = wx.StaticText(panel, label="K:", pos=(5, 200))
        self.text_ctrl_knn_k = wx.TextCtrl(panel, pos=(80, 200))

        self.static_text_knn_test = wx.StaticText(panel, label="Test:", pos=(5, 230))
        self.text_ctrl_knn_test = wx.TextCtrl(panel, pos=(80, 230))
        self.text_ctrl_knn_test.SetValue("0.3")

        self.radio_button_knn_euklid = wx.RadioButton(panel, 11, label='Euklidsko r.', pos=(5, 260), style=wx.RB_GROUP)
        self.radio_button_knn_menhetn = wx.RadioButton(panel, 22, label='Menhetn r.', pos=(115, 260))

        self.button_knn_ucitaj_podatke = wx.Button(panel, label='Ucitaj podatke', pos=(5, 290))
        self.button_knn_ucitaj_podatke.Bind(wx.EVT_BUTTON, self.import_data_knn_event)

        self.button_knn_knn = wx.Button(panel, label='KNN', pos=(115, 290))
        self.button_knn_knn.Bind(wx.EVT_BUTTON, self.knn_event)

        # PREDIKCIJA ###############################

        self.static_text_pred_predikcija_naslov = wx.StaticText(panel, label="PREDIKCIJA", pos=(500, 10))

        self.static_text_pred_kvadratura = wx.StaticText(panel, label="Kvadratura:", pos=(500, 40))
        self.text_ctrl_pred_kvadratura = wx.TextCtrl(panel, pos=(580, 40))

        self.static_text_pred_broj_soba = wx.StaticText(panel, label="Broj soba:", pos=(500, 70))
        self.text_ctrl_pred_broj_soba = wx.TextCtrl(panel, pos=(580, 70))

        self.static_text_pred_spratnost = wx.StaticText(panel, label="Spratnost:", pos=(500, 100))
        self.text_ctrl_pred_spratnost = wx.TextCtrl(panel, pos=(580, 100))

        self.static_text_pred_udaljenost_od_centra = wx.StaticText(panel, label="Koordinate:", pos=(500, 130))
        self.text_ctrl_pred_udaljenost_od_centra_x = wx.TextCtrl(panel, pos=(580, 130), size=(50, -1))
        self.text_ctrl_pred_udaljenost_od_centra_y = wx.TextCtrl(panel, pos=(640, 130), size=(50, -1))

        self.static_text_pred_tip_objekta = wx.StaticText(panel, label="Tip objekta:", pos=(500, 160))
        self.radio_pred_novo_g = wx.RadioButton(panel, 33, label='novogradnja', pos=(580, 160), style=wx.RB_GROUP)
        self.radio_pred_staro_g = wx.RadioButton(panel, 44, label='starogradnja', pos=(690, 160))
        self.radio_pred_ne_koristi_g = wx.RadioButton(panel, 55, label='ne koristi odliku', pos=(580, 180))
        self.radio_pred_ne_koristi_g.SetValue(True)

        self.static_text_pred_cena = wx.StaticText(panel, label="Cena:", pos=(500, 210))
        self.static_text_pred_cena_izlaz = wx.StaticText(panel, label="", pos=(580, 210))

        self.button_pred_lr = wx.Button(panel, label='Linearna regresija', pos=(590, 250))
        self.button_pred_lr.Bind(wx.EVT_BUTTON, self.lr_predict_event)

        self.button_pred_lr = wx.Button(panel, label='Treniraj', pos=(500, 250))
        self.button_pred_lr.Bind(wx.EVT_BUTTON, self.lr_train_event)

        self.static_text_pred_k_knn = wx.StaticText(panel, label="K:", pos=(500, 290))
        self.text_ctrl_pred_k_knn = wx.TextCtrl(panel, pos=(580, 290))

        self.button_pred_knn_ucitaj_podatke = wx.Button(panel, label='Ucitaj podatke', pos=(500, 320))
        self.button_pred_knn_ucitaj_podatke.Bind(wx.EVT_BUTTON, self.knn_pred_load_data_event)

        self.button_pred_knn = wx.Button(panel, label='KNN', pos=(610, 320))
        self.button_pred_knn.Bind(wx.EVT_BUTTON, self.knn_pred_event)

        # #############

        self.Show()

        self.lr_model = None
        self.lr_means = None
        self.lr_std = None
        self.knn_model = None
        self.lock = threading.Lock()

    def read_features(self) -> List[str]:
        x_features = []

        if self.check_box_kvadratura.IsChecked():
            x_features.append('kvadratura')
        if self.check_box_broj_soba.IsChecked():
            x_features.append('broj_soba')
        if self.check_box_spratnost.IsChecked():
            x_features.append('spratnost')
        if self.check_box_udaljenost_od_centra.IsChecked():
            x_features.append('udaljenost_od_centra')
        if self.check_box_tip_objekta.IsChecked():
            x_features.append('tip_objekta_klasa')

        if not x_features:
            x_features = X_FEATURE_LIST

        return x_features

    def read_features_for_prediction(self) -> List[str]:
        x_features = []

        if self.text_ctrl_pred_kvadratura.GetValue() != "":
            x_features.append(float(self.text_ctrl_pred_kvadratura.GetValue()))
        if self.text_ctrl_pred_broj_soba.GetValue() != "":
            x_features.append(float(self.text_ctrl_pred_broj_soba.GetValue()))
        if self.text_ctrl_pred_spratnost.GetValue() != "":
            x_features.append(float(self.text_ctrl_pred_spratnost.GetValue()))
        if self.text_ctrl_pred_udaljenost_od_centra_x.GetValue() != "" and \
                self.text_ctrl_pred_udaljenost_od_centra_y.GetValue() != "":
            distance = math.sqrt(
                pow(
                    (CENTER_OF_BELGRADE_X - float(self.text_ctrl_pred_udaljenost_od_centra_x.GetValue())),
                    2)
                + pow(
                    (CENTER_OF_BELGRADE_Y - float(self.text_ctrl_pred_udaljenost_od_centra_y.GetValue())),
                    2)
            )
            x_features.append(distance)
        if not self.radio_pred_ne_koristi_g.GetValue():
            if self.radio_pred_novo_g.GetValue():
                x_features.append(1)
            else:
                x_features.append(2)

        return x_features

    def lin_reg_event(self, event):
        access = self.lock.acquire(blocking=False)
        if access:
            thread = threading.Thread(target=self.lin_reg_thread)
            thread.start()

    def import_data_knn_event(self, event):
        access = self.lock.acquire(blocking=False)
        if access:
            thread = threading.Thread(target=self.import_data_knn_thread)
            thread.start()

    def knn_event(self, event):
        access = self.lock.acquire(blocking=False)
        if access:
            thread = threading.Thread(target=self.knn_thread)
            thread.start()

    def lr_train_event(self, event):
        access = self.lock.acquire(blocking=False)
        if access:
            thread = threading.Thread(target=self.lr_train_thread)
            thread.start()

    def lr_predict_event(self, event):
        access = self.lock.acquire(blocking=False)
        if access:
            thread = threading.Thread(target=self.lr_predict_thread)
            thread.start()

    def knn_pred_event(self, event):
        access = self.lock.acquire(blocking=False)
        if access:
            thread = threading.Thread(target=self.knn_pred_thread)
            thread.start()

    def knn_pred_load_data_event(self, event):
        access = self.lock.acquire(blocking=False)
        if access:
            thread = threading.Thread(target=self.knn_pred_load_data_thread)
            thread.start()

    def lr_train_thread(self):
        cls()

        print("Ucitavanje podataka...")
        x_features = self.read_features()
        x_values, y_values = load_data_from_csv(path="../data/filtered_data.csv", x_features=x_features)
        x_values = x_values.to_numpy()
        y_values = y_values.to_numpy()
        print("Ucitavanje podataka zavreseno.")

        print("Skaliranje odlika...")
        self.lr_means = []
        self.lr_std = []
        for i in range(x_values.shape[1]):
            self.lr_means.append(int(np.mean(x_values[:, i])))
            self.lr_std.append(np.std(x_values[:, i]))
            x_values[:, i] = (x_values[:, i] - int(np.mean(x_values[:, i]))) / np.std(x_values[:, i])

        # y_values = (y_values - int(np.mean(y_values))) / np.std(y_values)
        print("Skaliranje odlika zavrseno.")

        print("Deljenje podataka...")
        x_train, y_train, x_test, y_test = split_data_lr(x_values, y_values, 0.0)
        print("Deljenje podataka zavrseno.")

        print("Obucavanje...")
        regression = LinearRegressionPSZ(x_train, y_train)
        regression.train()
        self.lr_model = regression
        print("Obucavanje zavrseno.")

        self.lock.release()

    def lr_predict_thread(self):
        cls()

        print("Predikcija linearom regresijom zapoceta...")

        if not self.lr_model:
            print("Greska: Nije kreiran model linearne regresije.")
            self.lock.release()
            return

        x_values = self.read_features_for_prediction()

        for i in range(0, len(x_values)):
            value = x_values[i]
            scaled_value = (value - self.lr_means[i]) / self.lr_std[i]
            x_values[i] = scaled_value

        if len(x_values) != self.lr_model.num_of_features():
            print("Greska: Model nema isti broj odlika kao i uneti podatak.")
            self.lock.release()
            return

        value = self.lr_model.predict(np.array(x_values))
        self.static_text_pred_cena_izlaz.SetLabel(str(value[0]))

        print("Predikcija linearom regresijom zavresema.")
        self.lock.release()

    def knn_pred_load_data_thread(self):
        self.import_data_knn_thread(unlock=False, full_data_set=True)
        self.lock.release()

    def knn_pred_thread(self):
        cls()
        print("KNN predikcija zapoceta...")

        if not self.knn_model:
            print("Greska: Nije kreiran KNN model.")
            self.lock.release()
            return

        self.static_text_pred_cena_izlaz.SetLabel("")

        if self.radio_button_knn_euklid.GetValue():
            metric = DistanceMetric.euclidean_distance
        else:
            metric = DistanceMetric.manhattan_distance

        x_values = self.read_features_for_prediction()

        if len(x_values) != self.knn_model.num_of_features():
            print("Greska: Model nema isti broj odlika kao i uneti podatak.")
            self.lock.release()
            return

        pred_classes = self.knn_model.classify(pd.Series(x_values), metric=metric)
        pred_class = get_predicted_class(pred_classes)

        if pred_class == 0:
            self.static_text_pred_cena_izlaz.SetLabel("manja od 49 000")
        if pred_class == 1:
            self.static_text_pred_cena_izlaz.SetLabel("izmedju 50 000 i 99 999")
        if pred_class == 2:
            self.static_text_pred_cena_izlaz.SetLabel("izmedju 100 000 i 149 999")
        if pred_class == 3:
            self.static_text_pred_cena_izlaz.SetLabel("izmedju 150 000 i 199 999")
        if pred_class == 4:
            self.static_text_pred_cena_izlaz.SetLabel("200 000 ili visa")

        print("KNN predikcija zavrsena.")
        self.lock.release()

    def lin_reg_thread(self):
        cls()
        print("Linearna regresija zapoceta...")

        iterations = self.text_ctrl_lr_iteracije.GetValue()
        alpha = self.text_ctrl_lr_alfa.GetValue()
        test_size = self.text_ctrl_lr_test.GetValue()

        try:
            if iterations != '':
                iterations = int(iterations)
            else:
                iterations = 500
        except ValueError:
            print("Broj iteracija nije ceo broj.")
            self.lock.release()
            return

        try:
            if alpha != '':
                alpha = float(alpha)
            else:
                alpha = 0.00001
        except ValueError:
            print("Broj alfa nije broj u pokretnom zarezu.")
            self.lock.release()
            return

        try:
            if test_size != '':
                test_size = float(test_size)
            else:
                test_size = 0.3
        except ValueError:
            print("Procenat podataka za testiranje nije broj u pokretnom zarezu.")
            self.lock.release()
            return

        x_features = self.read_features()

        print("Ucitavanje podataka...")
        x_values, y_values = load_data_from_csv(path="../data/filtered_data.csv", x_features=x_features)
        x_values = x_values.to_numpy()
        y_values = y_values.to_numpy()
        print("Ucitavanje podataka zavreseno.")

        print("Skaliranje odlika...")
        for i in range(x_values.shape[1]):
            x_values[:, i] = (x_values[:, i] - int(np.mean(x_values[:, i]))) / np.std(x_values[:, i])

        y_values = (y_values - int(np.mean(y_values))) / np.std(y_values)
        print("Skaliranje odlika zavrseno.")

        print("Deljenje podataka...")
        x_train, y_train, x_test, y_test = split_data_lr(x_values, y_values, test_size)
        print("Deljenje podataka zavrseno.")

        print("Obucavanje...")
        regression = LinearRegressionPSZ(x_train, y_train, alpha=alpha, num_of_iter=iterations)
        train_cost, num_epochs = regression.train()
        _, RMSE_train = regression.test(x_train, y_train)
        print(f"RMSE (skup za obucavanje): {RMSE_train}")
        _, RMSE_test = regression.test(x_test, y_test)
        print(f"RMSE (skup za testiranje): {RMSE_test}")
        plot_error_function(train_cost, num_epochs)
        print("Obucavanje zavrseno.")

        print("Linearna regresija zavrsena.")
        self.lock.release()

    def import_data_knn_thread(self, unlock: bool = True, full_data_set: bool = False):
        cls()

        if full_data_set is False:
            test_size = self.text_ctrl_knn_test.GetValue()

            try:
                if test_size != '':
                    test_size = float(test_size)
                else:
                    test_size = 0.25
            except ValueError:
                print("Procenat podataka za testiranje nije broj u pokretnom zarezu.")
                self.lock.release()
                return
        else:
            test_size = 0.0

        x_features = self.read_features()

        print("Ucitavanje podataka...")
        x_values, y_values = load_data_from_csv(path="../data/filtered_data.csv", x_features=x_features)
        print("Ucitavanje podataka zavrseno.")

        for i in range(0, y_values.shape[0]):
            new_value: int = -1
            if y_values['cena'][i] <= 49999:
                new_value = 0
            if 50000 <= y_values['cena'][i] <= 99999:
                new_value = 1
            if 100000 <= y_values['cena'][i] <= 149999:
                new_value = 2
            if 150000 <= y_values['cena'][i] <= 199999:
                new_value = 3
            if y_values['cena'][i] >= 200000:
                new_value = 4
            y_values['cena'][i] = new_value

        features_with_class = x_values.copy(deep=True)
        features_with_class['klasa'] = y_values.copy(deep=True)

        print("Deljenje podataka...")
        train_data, test_data = split_data_knn(features_with_class, test_size)
        print("Deljenje podataka zavrseno.")
        train_data_x = train_data[x_features]
        train_data_y = train_data[['klasa']]
        test_data_x = test_data[x_features]
        test_data_y = test_data[['klasa']]

        self.train_data_x = train_data_x
        self.train_data_y = train_data_y
        self.test_data_x = test_data_x
        self.test_data_y = test_data_y

        if not full_data_set:
            print("Inicijalizovanje KNN algoritma...")
            self.knn_model = KNN(input_data=train_data_x, correct_output_class=train_data_y)
            print("Inicijalizovanje KNN algoritma zavrseno.")

            self.text_ctrl_knn_k.SetValue(str(self.knn_model.k()))
        else:
            print("Inicijalizovanje KNN algoritma...")
            self.knn_model = KNN(input_data=train_data_x, correct_output_class=train_data_y)
            print("Inicijalizovanje KNN algoritma zavrseno.")

            self.text_ctrl_pred_k_knn.SetValue(str(self.knn_model.k()))

        if unlock is True:
            self.lock.release()

    def knn_thread(self):
        cls()

        if self.radio_button_knn_euklid.GetValue():
            metric = DistanceMetric.euclidean_distance
        else:
            metric = DistanceMetric.manhattan_distance

        confusion_matrix = [[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]]
        print("Klase:")
        print("0 - cena <= 49 999")
        print("1 - 50 000 <= cena <= 99 999")
        print("2 - 100 000 <= cena <= 149 999")
        print("3 - 150 000 <= cena <= 199 999")
        print("4 - cena >= 200 000")
        print("Primena KNN algoritma na podatke iz skupa za testiranje...")
        for i in range(0, self.test_data_x.shape[0]):
            pred_classes = self.knn_model.classify(self.test_data_x.iloc[i], metric=metric)
            pred_class = get_predicted_class(pred_classes)
            true_class = self.test_data_y.iloc[i].values.tolist()[0]
            confusion_matrix[pred_class][true_class] = confusion_matrix[pred_class][true_class] + 1
            # print(f"{self.test_data_x.iloc[i].values.tolist()} pripada klasi {true_class}. Prediktovana klasa {pred_class}.")
            print(f"Podatak pripada klasi {true_class}. Prediktovana klasa {pred_class}.")
        print("Primena KNN algoritma na podatke iz skupa za testiranje zavrsena.")

        accuracy = calculate_accuracy(confusion_matrix)
        print(f"Preciznost (accuracy): {accuracy}")

        self.lock.release()


if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame()
    app.MainLoop()
