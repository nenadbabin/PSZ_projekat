from typing import List

import numpy as np
import wx
import os

from k_nearest_neighbors.KNN import KNN, DistanceMetric
from k_nearest_neighbors.knn_utility import split_data_knn, get_predicted_class, calculate_accuracy
from linear_regression.lin_reg import MultipleLinearRegression
from linear_regression.linear_regression_utility import split_data_lr
from utility.helpers import load_data, X_FEATURE_LIST


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


class MyFrame(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title='PSZ', size=(440, 400))
        panel = wx.Panel(self)

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

        self.static_text_lr_iteracije = wx.StaticText(panel, label="LINEARNA REGRESIJA", pos=(5, 10))

        self.static_text_lr_iteracije = wx.StaticText(panel, label="Broj iteracija:", pos=(5, 40))
        self.text_ctrl_lr_iteracije = wx.TextCtrl(panel, pos=(80, 40))

        self.static_text_lr_alfa = wx.StaticText(panel, label="Alfa:", pos=(5, 70))
        self.text_ctrl_lr_alfa = wx.TextCtrl(panel, pos=(80, 70))

        self.static_text_lr_test = wx.StaticText(panel, label="Test:", pos=(5, 100))
        self.text_ctrl_lr_test = wx.TextCtrl(panel, pos=(80, 100))

        self.button_lr_treniraj = wx.Button(panel, label='Linearna regresija', pos=(5, 130))
        self.button_lr_treniraj.Bind(wx.EVT_BUTTON, self.lin_reg_event)

        self.static_text_knn = wx.StaticText(panel, label="K NAJBLIZIH SUSEDA junk", pos=(5, 170))

        self.static_text_knn_k = wx.StaticText(panel, label="K:", pos=(5, 200))
        self.text_ctrl_knn_k = wx.TextCtrl(panel, pos=(80, 200))

        self.static_text_knn_test = wx.StaticText(panel, label="Test:", pos=(5, 230))
        self.text_ctrl_knn_test = wx.TextCtrl(panel, pos=(80, 230))

        self.radio_button_knn_euklid = wx.RadioButton(panel, 11, label='Euklidsko r.', pos=(5, 260), style=wx.RB_GROUP)
        self.radio_button_knn_menhetn = wx.RadioButton(panel, 22, label='Menhetn r.', pos=(115, 260))

        self.button_knn_ucitaj_podatke = wx.Button(panel, label='Ucitaj podatke', pos=(5, 290))
        self.button_knn_ucitaj_podatke.Bind(wx.EVT_BUTTON, self.import_data_knn)

        self.button_knn_knn = wx.Button(panel, label='KNN', pos=(115, 290))
        self.button_knn_knn.Bind(wx.EVT_BUTTON, self.knn_event)

        self.Show()

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

    def lin_reg_event(self, event):
        cls()

        iterations = self.text_ctrl_lr_iteracije.GetValue()
        alpha = self.text_ctrl_lr_alfa.GetValue()
        test_size = self.text_ctrl_lr_test.GetValue()

        try:
            if iterations != '':
                iterations = int(iterations)
            else:
                iterations = 200
        except ValueError:
            print("Broj iteracija nije ceo broj.")
            return

        try:
            if alpha != '':
                alpha = float(alpha)
            else:
                alpha = 0.00001
        except ValueError:
            print("Broj alfa nije broj u pokretnom zarezu.")
            return

        try:
            if test_size != '':
                test_size = float(test_size)
            else:
                test_size = 0.25
        except ValueError:
            print("Procenat podataka za testiranje nije broj u pokretnom zarezu.")
            return

        x_features = self.read_features()

        x_values, y_values = load_data(x_features)
        x_values = x_values.to_numpy()
        y_values = y_values.to_numpy()

        for i in range(x_values.shape[1]):
            x_values[:, i] = (x_values[:, i] - int(np.mean(x_values[:, i]))) / np.std(x_values[:, i])

        x_train, y_train, x_test, y_test = split_data_lr(x_values, y_values, test_size)

        regression = MultipleLinearRegression()
        weights_trained, train_loss, num_epochs = regression.train(x_train, y_train, epochs=iterations, alpha=alpha)
        test_pred, test_loss = regression.test(x_test, y_test, weights_trained)
        regression.plotLoss(train_loss, num_epochs)

    def import_data_knn(self, event):
        cls()
        test_size = self.text_ctrl_knn_test.GetValue()

        try:
            if test_size != '':
                test_size = float(test_size)
            else:
                test_size = 0.25
        except ValueError:
            print("Procenat podataka za testiranje nije broj u pokretnom zarezu.")
            return

        x_features = self.read_features()

        x_values, y_values = load_data()

        for i in range(0, y_values.shape[0]):
            new_value: int = -1
            if y_values['cena'][i] <= 49999:
                new_value = 0
            if 50000 <= y_values['cena'][i] <= 99999:
                new_value = 1
            if 100000 <= y_values['cena'][i] <= 149000:
                new_value = 2
            if 150000 <= y_values['cena'][i] <= 199000:
                new_value = 3
            if y_values['cena'][i] >= 200000:
                new_value = 4
            y_values['cena'][i] = new_value

        features_with_class = x_values.copy(deep=True)
        features_with_class['klasa'] = y_values.copy(deep=True)

        train_data, test_data = split_data_knn(features_with_class)
        train_data_x = train_data[x_features]
        train_data_y = train_data[['klasa']]
        test_data_x = test_data[x_features]
        test_data_y = test_data[['klasa']]

        self.train_data_x = train_data_x
        self.train_data_y = train_data_y
        self.test_data_x = test_data_x
        self.test_data_y = test_data_y
        self.knn = KNN(input_data=train_data_x, correct_output_class=train_data_y)

        self.text_ctrl_knn_k.SetValue(str(self.knn.k))

    def knn_event(self, event):

        if self.radio_button_knn_euklid.GetValue():
            metric = DistanceMetric.euclidean_distance
        else:
            metric = DistanceMetric.manhattan_distance

        confusion_matrix = [[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]]

        for i in range(0, self.test_data_x.shape[0]):
            pred_classes = self.knn.classify(self.test_data_x.iloc[i], metric=metric)
            pred_class = get_predicted_class(pred_classes)
            true_class = self.test_data_y.iloc[i].values.tolist()[0]
            confusion_matrix[pred_class][true_class] = confusion_matrix[pred_class][true_class] + 1
            print(f"{self.test_data_x.iloc[i].values.tolist()} belongs to class {true_class}. Predicted class {pred_class}.")

        accuracy = calculate_accuracy(confusion_matrix)
        print(f"Accuracy: {accuracy}")


if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame()
    app.MainLoop()
