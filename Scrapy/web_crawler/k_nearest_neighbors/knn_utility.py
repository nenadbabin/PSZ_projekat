from typing import List


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
