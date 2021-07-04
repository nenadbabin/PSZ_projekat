from k_nearest_neighbors.KNN import KNN, DistanceMetric
from k_nearest_neighbors.knn_utility import get_predicted_class, calculate_accuracy, split_data_knn
from utility.helpers import load_data, X_FEATURE_LIST, Y_FEATURE_LIST, load_data_from_csv, plot
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    x_values, y_values = load_data_from_csv(path="../data/filtered_data.csv")

    for i in range(0, y_values.shape[0]):
        new_value: int = -1
        if y_values['cena'][i] <= 49999:
            new_value = 0
        elif 50000 <= y_values['cena'][i] <= 99999:
            new_value = 1
        elif 100000 <= y_values['cena'][i] <= 149999:
            new_value = 2
        elif 150000 <= y_values['cena'][i] <= 199999:
            new_value = 3
        elif y_values['cena'][i] >= 200000:
            new_value = 4
        y_values['cena'][i] = new_value

    features_with_class = x_values.copy(deep=True)
    features_with_class['klasa'] = y_values.copy(deep=True)

    plot(features_with_class['kvadratura'], features_with_class['klasa'], "Raspodela kvadratura-cena", "kvadratura", "klasa")
    plot(features_with_class['broj_soba'], features_with_class['klasa'], "Raspodela broj_soba-cena", "broj_soba", "klasa")
    plot(features_with_class['spratnost'], features_with_class['klasa'], "Raspodela spratnost-cena", "spratnost", "klasa")
    plot(features_with_class['udaljenost_od_centra'], features_with_class['klasa'], "Raspodela udaljenost_od_centra-klasa",
         "udaljenost_od_centra", "klasa")
    plot(features_with_class['tip_objekta_klasa'], features_with_class['klasa'], "Raspodela tip_objekta_klasa-cena", "tip_objekta_klasa",
         "cena")

    train_data, test_data = split_data_knn(features_with_class)
    train_data_x = train_data[X_FEATURE_LIST]
    train_data_y = train_data[['klasa']]
    test_data_x = test_data[X_FEATURE_LIST]
    test_data_y = test_data[['klasa']]

    # x_train, x_test, y_train, y_test = \
    #     train_test_split(x_values, y_values, test_size=0.25, random_state=42, stratify=y_values)

    knn = KNN(input_data=train_data_x, correct_output_class=train_data_y)

    confusion_matrix = [[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]]

    for i in range(0, test_data_x.shape[0]):
        pred_classes = knn.classify(test_data_x.iloc[i], metric=DistanceMetric.euclidean_distance)
        pred_class = get_predicted_class(pred_classes)
        true_class = test_data_y.iloc[i].values.tolist()[0]
        confusion_matrix[pred_class][true_class] = confusion_matrix[pred_class][true_class] + 1
        print(f"{test_data_x.iloc[i].values.tolist()} belongs to class {true_class}. Predicted class {pred_class}.")

    accuracy = calculate_accuracy(confusion_matrix)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
