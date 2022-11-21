import sys
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from typing import List
from c45 import C45Tree
from utils import get_random_selection

warnings.filterwarnings("ignore")


def solve(tree: C45Tree, test: pd.DataFrame) -> tuple:
    test_results = []
    predict_positive = set()
    real_positive = set()
    predict_negative = set()
    real_negative = set()
    for i in range(test.shape[0]):
        predict = tree.traverse(test.drop(columns='GRADE').iloc[i])
        if test.iloc[i].GRADE == 1:
            real_positive.add(i)
        else:
            real_negative.add(i)
        test_results.append(predict)
        if predict == 1:
            predict_positive.add(i)
        else:
            predict_negative.add(i)
    return predict_negative, real_negative, predict_positive, real_positive, test_results


def print_result(predict_negative: set,
                 real_negative: set,
                 predict_positive: set,
                 real_positive: set,
                 test_results: List[np.int64]):
    true_positive = predict_positive & real_positive
    true_negative = predict_negative & real_negative

    accuracy = (len(true_positive) + len(true_negative)) / len(test_results)

    print('Accuracy :', accuracy)

    print(f'Precision для успевающих: {len(true_positive) / len(predict_positive)}')
    print(f'Recall для успевающих: {len(true_positive) / len(real_positive)}')

    print(f'Precision для неуспевающих: {len(true_negative) / len(predict_negative)}')
    print(f'Recall для неуспевающих: {len(true_negative) / len(real_negative)}')


def draw_graph(test: pd.DataFrame, test_results: list):
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(test.GRADE.tolist(), test_results)
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC-ROC')
    plt.savefig('./graphs/ROC.png')
    plt.clf()

    precision, recall, _ = precision_recall_curve(test.GRADE.tolist(), test_results)
    plt.title('AUC-PR')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall, precision, lw=2)
    plt.plot([0, 1], [1, 0], linestyle='--')
    plt.savefig('./graphs/PR.png')


def main() -> None:
    data = get_random_selection(pd.read_csv('DATA.csv', index_col=0))
    data['GRADE'] = (data.GRADE >= 3).astype(int)

    print(data)
    train, test = train_test_split(data)
    tree = C45Tree(train)

    predict_negative, real_negative, predict_positive, real_positive, test_results = solve(tree, test)

    print_result(predict_negative, real_negative, predict_positive, real_positive, test_results)

    draw_graph(test, test_results)


if __name__ == '__main__':
    main()
