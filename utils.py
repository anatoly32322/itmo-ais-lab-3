from math import log

import pandas as pd


def get_random_selection(selection: pd.DataFrame) -> pd.DataFrame:
    return selection.sample(frac=0.5)


def freq(data_part, cj):
    return (data_part.GRADE == cj).sum()


def info(data_part):
    sum = 0
    for class_value in data_part.GRADE.unique():
        p = freq(data_part, class_value) / data_part.shape[0]
        sum += p * log(p, 2)
    return -sum


def info_x(data_part, partition_field):
    sum = 0
    for field_value in data_part[partition_field].unique():
        subset = data_part[data_part[partition_field] == field_value]
        sum += subset.shape[0] * info(subset) / data_part.shape[0]
    return sum


def split_info_x(data_part, partition_field):
    sum = 0
    for field_value in data_part[partition_field].unique():
        subset = data_part[data_part[partition_field] == field_value]
        p = subset.shape[0] / data_part.shape[0]
        sum += p * log(p, 2)
    return -sum
