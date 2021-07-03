from k_nearest_neighbors.KNN import KNN, DistanceMetric
from k_nearest_neighbors.knn_utility import get_predicted_class, calculate_accuracy, split_data_knn
from utility.helpers import load_data, X_FEATURE_LIST, Y_FEATURE_LIST, load_data_from_csv
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    x_values, y_values = load_data_from_csv(path="../data/filtered_data.csv")

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
