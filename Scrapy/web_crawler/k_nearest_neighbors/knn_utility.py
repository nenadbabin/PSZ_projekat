from typing import List

import pandas as pd
from pandas import DataFrame


def get_predicted_class(classes: dict) -> int:
    number_of_instances = []
    for i in range(0, 5):
        try:
            number_of_instances.append(classes[i])
        except KeyError:
            number_of_instances.append(0)

    max_instances = 0
    index = 0

    for i in range(0, 5):
        if number_of_instances[i] > max_instances:
            max_instances = number_of_instances[i]
            index = i

    return index


def split_data(data: DataFrame, split_ratio: float = 0.25):

    shuffled_data = data.sample(frac=1).reset_index(drop=True)

    mask = shuffled_data['klasa'] == 0
    class_0 = shuffled_data[mask]
    number_of_data = class_0.shape[0]
    test_num = int(number_of_data * split_ratio)
    train_num = number_of_data - test_num

    test_data = class_0.head(test_num).copy(deep=True)
    train_data = class_0.tail(train_num).copy(deep=True)

    for i in range(1, 5):
        mask = shuffled_data['klasa'] == i
        specific_data_class = shuffled_data[mask]
        number_of_data = specific_data_class.shape[0]
        test_num = int(number_of_data * split_ratio)
        train_num = number_of_data - test_num

        test_data_class = specific_data_class.head(test_num).copy(deep=True)
        train_data_class = specific_data_class.tail(train_num).copy(deep=True)

        # train_data.append(train_data_class, ignore_index=True)
        # test_data.append(test_data_class, ignore_index=True)

        train_data = pd.concat([train_data, train_data_class], ignore_index=True)
        test_data = pd.concat([test_data, test_data_class], ignore_index=True)

    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)

    return train_data, test_data


def calculate_accuracy(confusion_matrix: List[List[int]]) -> float:
    """
    Calculates Accuracy, Pmicro, Rmicro, Fmicro.
    """
    true_positives: int = 0
    all_values: int = 0
    for i in range(0, 5):
        true_positives += confusion_matrix[i][i]

    for i in range(0, 5):
        for j in range(0, 5):
            all_values += confusion_matrix[i][j]

    return true_positives / all_values
