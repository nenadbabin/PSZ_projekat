from k_nearest_neighbors.KNN import KNN, DistanceMetric
from k_nearest_neighbors.knn_utility import get_predicted_class, calculate_accuracy
from utility.helpers import load_data
from sklearn.model_selection import train_test_split


def main():
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

    x_train, x_test, y_train, y_test = \
        train_test_split(x_values, y_values, test_size=0.25, random_state=42, stratify=y_values)
    knn = KNN(input_data=x_train, correct_output_class=y_train)

    confusion_matrix = [[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]]

    for i in range(0, x_test.shape[0]):
        pred_classes = knn.classify(x_test.iloc[i], metric=DistanceMetric.euclidean_distance)
        pred_class = get_predicted_class(pred_classes)
        true_class = y_test.iloc[i].values.tolist()[0]
        confusion_matrix[pred_class][true_class] = confusion_matrix[pred_class][true_class] + 1
        print(f"{x_test.iloc[i].values.tolist()} belongs to class {true_class}. Predicted class {pred_class}.")

    accuracy = calculate_accuracy(confusion_matrix)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
