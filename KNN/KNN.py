# coding=utf8
import numpy as np
from functools import partial

class KNN(object):
    '''
    K近邻算法主体类-最朴素KNN
    '''

    Metrics = {
        'euclidean': 2, # 欧式距离
        'manhattan': 1, # 曼哈顿距离
        'chebyshev': np.inf, # 切比雪夫距离
    }

    def __init__(self, k=2, metric='euclidean', p=4):
        '''
        选择前k个最近的特征点用于统计类别
        metric: 距离度量指标. 包含
        'euclidean':  欧式距离
        'manhattan':  曼哈顿距离
        'chebyshev':  切比雪夫距离
        'minkowski':  闵可夫斯基距离, 此距离必须给定参数p
        '''
        # 记录k值
        self.k = k
        # 选择距离度量指标
        metric = metric.lower()
        if metric == 'minkowski':
            self.distance = partial(np.linalg.norm, ord=int(p), axis=-1)
        else:
            ord = self.Metrics.get(metric, self.Metrics['euclidean'])
            self.distance = partial(np.linalg.norm, ord=ord, axis=-1)
    
    def fit(self, x, y):
        '''
        只实现了暴力计算的方法, 在训练阶段只需记录训练集即可
        还可以使用KD树对训练进行记录, 改进部分参见sklearn
        '''
        self.x, self.y = np.asarray(x), np.asarray(y)
    
    def predict(self, x):
        '''
        根据训练集选择最近的k个训练样本, 并从中选择出现概率最高的标签
        '''
        if np.ndim(x) < 2:
            x = np.asarray([x])
        # 计算距离矩阵
        dis_mat = np.array([self.distance(xi - self.x) for xi in x])
        # 每个样本取前k个最小值所对应的类别标签
        y = self.y[dis_mat.argsort(axis=-1)[:, :self.k]]
        # 为每个样本yi投票选择出现概率最大的类别u[max]
        return [u[cnts.argmax()] for u, cnts in (np.unique(yi, return_counts=True) for yi in y)]

    def visualize_2d(self, x, y):
        '''
        可视化样本训练集, 以及训练数据
        '''
        from matplotlib import pyplot as plt
        plt.xlabel('X1')
        plt.ylabel('X2')
        # 获得训练集正反例并绘制散点图
        pos, neg = self.y > 0, self.y < 0 # 正反例掩码
        plt.plot(self.x[pos, 0], self.x[pos, 1], 'ro') # 训练集正例用红色点标出
        plt.plot(self.x[neg, 0], self.x[neg, 1], 'go') # 训练集反例用蓝色点标出
        # 测试集正反例
        pos, neg = y > 0, y < 0 # 正反例掩码
        plt.plot(x[pos, 0], x[pos, 1], 'bo') # 测试集正例用红色点标出
        plt.plot(x[neg, 0], x[neg, 1], 'ko') # 测试集反例用蓝色点标出
        # 绘制图例
        plt.legend(['positive training samples', 'negative training samples', 'positive testing samples', 'negative testing samples'])
        plt.show()