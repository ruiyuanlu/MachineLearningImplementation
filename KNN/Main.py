# coding=utf8

from KNN import KNN
import numpy as np
np.random.seed(2018)


def load_data(path):
    arr = np.loadtxt(path)
    x, y = arr[:, :2], arr[:, -1]
    return x, y

if __name__ == '__main__':
    x, y = load_data('testSet.txt')
    train_cnt = 90
    train_x, train_y = x[:train_cnt], y[:train_cnt]
    test_x, test_y = x[train_cnt:], y[train_cnt:]

    knn = KNN(k=2)
    knn.fit(train_x, train_y)
    predict_y = knn.predict(test_x)
    # 输出精度
    acc = 1 - np.count_nonzero(predict_y - test_y) / len(test_y)
    print('精度:', acc)
    knn.visualize_2d(test_x, test_y)