import numpy as np
import pandas as pd

from utils import info, info_x, split_info_x


class C45Tree:
    leaf: bool
    partition_field: str
    leaf_class: np.ndarray
    subtrees: dict = {}
    most_popular_class: np.int64

    def __init__(self, data: pd.DataFrame) -> None:
        subset_fields = data.drop(columns='GRADE').columns

        subset_fields_gain_ratios = pd.Series(
            [gain_ratio(data, subset_field) * int(len(data[subset_field].unique()) != 1)
             for subset_field in subset_fields]).fillna(0)

        self.most_popular_class = data.GRADE.mode()[0]

        self.subtrees = {}

        self.leaf_class = data.GRADE.unique()

        if not any(subset_fields_gain_ratios) or len(data.GRADE.unique()) == 1:
            self.leaf = True
            return

        self.partition_field = subset_fields[np.argmax(
            subset_fields_gain_ratios)]
        self.leaf = False
        for partition_field_value in data[self.partition_field].unique():
            self.subtrees[partition_field_value] = C45Tree(
                data[data[self.partition_field] == partition_field_value].drop(columns=self.partition_field))

    def traverse(self, data_item: pd.DataFrame) -> np.int64:
        if not self.leaf:
            partition_value = data_item[self.partition_field]
            if partition_value in self.subtrees:
                return self.subtrees[partition_value].traverse(data_item)
            return self.most_popular_class
        final_class = self.leaf_class
        if len(final_class) == 1:
            return final_class[0]
        return self.most_popular_class


def gain_ratio(data: pd.DataFrame, partition_field: pd.Index):
    return (info(data) - info_x(data, partition_field)) / split_info_x(data, partition_field)
