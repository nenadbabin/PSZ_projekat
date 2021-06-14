from k_nearest_neighbors.KNN import KNN, DistanceMetric
from utility.helpers import load_data


def main():
    x_values, y_values = load_data()

    for i in range(0, y_values.shape[0]):
        new_value: int = 0
        if y_values['cena'][i] <= 49000:
            new_value = 1
        if 50000 <= y_values['cena'][i] <= 99999:
            new_value = 2
        if 100000 <= y_values['cena'][i] <= 149000:
            new_value = 3
        if 150000 <= y_values['cena'][i] <= 199000:
            new_value = 4
        if y_values['cena'][i] >= 200000:
            new_value = 5
        y_values['cena'][i] = new_value

    knn = KNN(input_data=x_values, correct_output_class=y_values)
    pred_class = knn.classify(x_values.iloc[3], metric=DistanceMetric.manhattan_distance)
    print(pred_class)


if __name__ == "__main__":
    main()
