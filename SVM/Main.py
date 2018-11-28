# coding=utf8

from SVM import SVM
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
    # svm = SVM()
    # svm = SVM('poly')
    # svm = SVM('poly', gamma=0.01, d=3, r=0.5)
    # svm = SVM('sigmoid')
    # svm = SVM('sigmoid', gamma=0.001, r=0)
    svm = SVM('rbf', gamma=1e-2) # rbf核函数似乎有问题？
    svm = SVM('rbf', gamma=1) # rbf核函数似乎有问题？
    # svm.fit(x, y, C=1, epsilon=1e-4, max_iter=40)
    svm.fit(train_x, train_y, C=1, epsilon=1e-4, max_iter=40)
    predict = svm.predict_class(test_x)
    print(predict)
    print(y)
    acc = 1 - np.count_nonzero(predict - test_y) / len(test_y)
    print('精度:', acc)
    svm.visualize_2d(x, y)