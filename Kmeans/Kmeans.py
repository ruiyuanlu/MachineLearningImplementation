# coding=utf8
from functools import partial
import numpy as np

class Kmeans(object):
    '''
    K均值聚类算法主体类-最朴素Kmeans
    '''
    Metrics = {
        'euclidean': 2, # 欧式距离
        'manhattan': 1, # 曼哈顿距离
        'chebyshev': np.inf, # 切比雪夫距离
    }

    def __init__(self, k=2, metric='euclidean', p=4):
        '''
        记录中心的数目k
        metric: 距离度量指标. 包含
        'euclidean':  欧式距离
        'manhattan':  曼哈顿距离
        'chebyshev':  切比雪夫距离
        'minkowski':  闵可夫斯基距离, 此距离必须给定参数p
        '''
        self.k = k
        # 选择距离度量指标
        metric = metric.lower()
        if metric == 'minkowski':
            self.distance = partial(np.linalg.norm, ord=int(p), axis=-1)
        else:
            ord = self.Metrics.get(metric, self.Metrics['euclidean'])
            self.distance = partial(np.linalg.norm, ord=ord, axis=-1)

    def fit(self, x, max_iter=40, eps=1e-5):
        '''
        无标签聚类, 当中心点不再变化时, 记录所有中心点并退出
        '''
        self.x = np.asarray(x)
        # 随机选择k个初始点
        idx = np.arange(len(x), dtype=int)
        np.random.shuffle(idx)
        points = self.x[idx[:self.k]] # 初始点矩阵
        # 迭代更新中心点
        for _ in range(max_iter):
            new_points = self.compute_points(points)
            if np.sum(abs(points - new_points)) < eps:
                break
            points = new_points # 更新中心点
        self.points = points

    def predict(self, x):
        '''
        预测与样本x中最接近的点的坐标
        '''
        return self._predict_(x, self.points)
    
    def visualize_2d(self, x, y):
        '''
        绘制训练集, 测试集以及聚类中心
        '''
        from matplotlib import pyplot as plt
        plt.xlabel('X1')
        plt.ylabel('X2')
        # 获得训练集正反例并绘制散点图
        pos, neg = y > 0, y < 0 # 正反例掩码
        plt.plot(x[pos, 0], x[pos, 1], 'ro') # 测试集正例用红色点标出
        plt.plot(x[neg, 0], x[neg, 1], 'go') # 测试集反例用蓝色点标出
        # 绘制聚类中心
        plt.plot(self.points[0:, 0], self.points[0:, 1], 'ko') # 黑色标出聚类中心1
        plt.plot(self.points[1:, 0], self.points[1:, 1], 'bo') # 蓝色标出聚类中心2
        # 绘制图例
        plt.legend(['positive samples', 'negative samples', 'center 1', 'center 2'])
        plt.show()

    def compute_points(self, points):
        '''
        根据当前中心点计算新的中心点
        '''
        # 为每个样本选择最近的中心点, 获得中心点索引 nearest_idx
        nearest_idx = self._predict_(self.x, points)
        # 为每类样本计算新的中心点
        return np.array([np.mean(self.x[nearest_idx == i], axis=0) for i in range(self.k)])

    def _predict_(self, x, points):
        '''
        计算样本点到中心点的距离, 并给出每个样本点的类别
        '''
        # 计算每个中心点到所有样本点距离矩阵, x 为样本矩阵, p 为中心点向量
        dis_mat = np.array([self.distance(p - x) for p in points])
        # 为每个样本选择最近的中心点, 计算每个样本对应的中心点索引
        return dis_mat.argmin(axis=0)