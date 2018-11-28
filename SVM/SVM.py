# coding=utf8
import numpy as np
from Kernel import kernel

class SVM(object):
    '''
    SVM算法主体类
    '''

    def __init__(self, kernel_name='linear', **kernel_params):
        '''
        SVM 构造函数
        '''
        self.kernel_name = kernel_name
        self.kernel_params = kernel_params
    
    def fit(self, x, y, C=1, epsilon=1e-4, max_iter=1000):
        '''
        SVM 训练函数
        '''
        x, y = np.asarray(x), np.asarray(y)
        # 初始化SVM分类器
        self._init_classifier_(x, y, C, epsilon)
        # 调用SMO算法进行分类
        self.SMO(x, y, C, epsilon, max_iter)
    
    def predict(self, x):
        '''
        对于新的样本矩阵x进行预测, 输出预测值结果
        '''
        if np.ndim(x) < 2:
            x = np.asarray([x])
        return np.array([self.compute_fx(sample) for sample in x])
    
    def predict_class(self, x):
        '''
        预测样本类别
        '''
        return np.sign(self.predict(x))
    
    def visualize_2d(self, x, y):
        '''
        可视化任意分类平面
        由SVM优化目标可知, 使得分类函数 fx == 0 的点为分类平面上的点,
        fx > 0 的为正类, fx < 0 的为反类
        fx = y * (wx + b), 由 compute_fx 给出
        '''
        from matplotlib import pyplot as plt
        plt.xlabel('X1')
        plt.ylabel('X2')
        # 获得正反例并绘制散点图
        pos, neg = y > 0, y < 0 # 正反例掩码
        plt.plot(x[pos, 0], x[pos, 1], 'ro') # 正例用红色点标出
        plt.plot(x[neg, 0], x[neg, 1], 'go') # 反例用蓝色点标出
        # 绘制支持向量
        non_zero_idx = np.nonzero(self.alpha)
        support_vector = x[non_zero_idx] # 由KKT条件, alpha > 0 的向量为支持向量
        plt.plot(support_vector[:, 0], support_vector[:, 1], 'bo')
        # 绘制分类边界
        low_x1, high_x1 = min(x[:, 0]) - 5, max(x[:, 0]) + 5 # 计算x轴y轴的上下界
        low_x2, high_x2 = min(x[:, 1]) - 5, max(x[:, 1]) + 5
        points = [(x0, y0) for x0 in np.arange(low_x1, high_x1, 0.1) for y0 in np.arange(low_x2, high_x2, 0.1)] # 构造网格点坐标
        points.sort(key=lambda p: abs(self.predict(p))) # 根据每个点到分类平面的距离排序
        x1, x2 = zip(*sorted(points[:100])) # 取最接近分类平面的100个点用于绘制分类边界
        plt.plot(x1, x2, 'o') # 绘制分类边界
        plt.legend(['positive samples', 'negative samples', 'support vectors', 'decision boundary'])
        plt.show()

    def _init_classifier_(self, x, y, C, epsilon):
        '''
        初始化SVM分类器, 在fit函数中调用
        Args:
            x: ndarray (N, M). 训练数据, 每一行为一个特征向量, N 为样本数量, M 为特征维度
            y: ndarray (N) or (1, N). 训练标签, 每一行为一个标签
            C: float 优化目标中的常数C
            epsilon: float 有些实现中也写作 torlerance, 允许的精度误差
            N: int 训练集的样本数量
            alpha: ndarray (N). 拉格朗日乘子矩阵, 目标函数的参数矩阵
            b: float 目标函数的截距
            E: ndarray (N). 误差 E = fx - y
            updated_E: ndarray (N). 记录E是否更新, True 为已更新, False 为未更新
        '''
        self.kernel_params['y'] = y
        self.kernel = kernel(self.kernel_name, **self.kernel_params)
        self.x, self.y = x, y
        self.C = C
        self.epsilon = epsilon
        self.N = x.shape[0]
        self.alpha = np.zeros(self.N)
        self.b = 0.0
        self.E = np.zeros(self.N)
        self.updated_E = np.zeros(self.N, dtype=bool)
    
    def SMO(self, x, y, C, epsilon, max_iter):
        '''
        SMO迭代优化算法
        Args:
            x: ndarray 训练数据, 每一行为一个特征向量
            y: ndarray 训练标签, 每一行为一个标签
            C: float 优化目标中的常数C
            epsilon: float 有些实现中也写作 torlerance, 允许的精度误差
            max_iter: int 最大迭代次数
        '''
        entire = True # 是否遍历全部数据, (false则只遍历非边界数据)
        for _ in range(max_iter): # 迭代算法
            # 计算需要遍历的样本索引集合, entire=True 遍历全部样本, 否则遍历非边界样本
            idx = range(self.N) if entire else self.find_nonbound(self.alpha, self.C)
            # 计算被更新的 (αi, αj) 总数目
            alpha_pair_changed_cnt = sum(map(self.inner_loop, idx))
            if alpha_pair_changed_cnt == 0:
                if entire: # 如果是在遍历所有样本情况下, (αi, αj)没有更新, 则退出
                    break
                else: # 如果是非边界样本遍历后未更新, 下一次迭代遍历全部样本
                    entire = True
            elif entire: # 在(αi, αj)有更新且遍历了全部样本的情况下, 下一次迭代只需遍历非边界样本
                entire = False

    def find_nonbound(self, alpha, C):
        '''
        根据拉格朗日因子alpha与惩罚因子C的关系
        给出非边界点的索引值
        alpha[i] = 0 或 c 时，称为边界点；0 < alpha[i] < c，称为非边界点（nonbound）
        '''
        return (i for i, a in enumerate(alpha) if 0 < a < C)

    def inner_loop(self, i):
        '''
        SVM 迭代算法内循环, 返回1表示更新了一对参数, 0表示未更新参数
        i: 对应于 alpha_i, 及其计算出的 E_i
        j: 对应于 alpha_j, 及其计算出的 E_j
        '''
        Ei = self.compute_E(i)
        if self.violate_KKT(i, Ei):
            self.update_E(i, Ei) # 更新 Ei, 用于提前return 0时
            j, Ej = self.select_j_by_i(i, Ei) # 根据alpha_i 选择 alpha_j
            low, high = self.compute_alpha_bounds(i, j) # 计算alpha的上下界
            if low < high: # low >= high, 则alpha为空集
                eta = self.compute_eta(i, j) # 如果非空, 计算eta, 为更新alpha准备
                if eta > 0:
                    # 更新alpha_j, alpha_j 需要裁剪
                    old_alpha_j = self.alpha[j]
                    new_alpha_j = self.compute_alpha_j(j, Ei, Ej, eta)
                    self.alpha[j] = self.clip_alpha(new_alpha_j, low, high) # 裁剪 alpha_j
                    self.update_E(j, self.compute_E(j)) # 计算并更新 Ej
                    # 更新alpha_i, alpha_i 不需裁剪, 只需根据求得的alpha_j求出
                    old_alpha_i = self.alpha[i]
                    self.alpha[i] = self.compute_alpha_i(i, j, old_alpha_j)
                    self.update_E(i, self.compute_E(i)) # 计算并更新 Ei
                    self.b = self.compute_b(i, j, Ei, Ej, old_alpha_i, old_alpha_j) # 更新b
                    return 1
        return 0

    def compute_E(self, k):
        '''
        计算样本 xk 的SVM目标函数误差 Ek
        '''
        return self.compute_fx(self.x[k]) - self.y[k]
    
    def update_E(self, i, Ei):
        '''
        更细Ei, 并记录状态为已更新 (True)
        '''
        self.updated_E[i], self.E[i] = True, Ei

    def compute_fx(self, xk):
        '''
        计算分类函数 f(x) = y * (wx + b) 的值, xk只能传入向量值
        w = sum(y * alpha * <xk, X>), b 为截距. 这里使用dot替代求和操作
        '''
        return self.alpha.dot(self.y * self.kernel(xk, self.x)) + self.b

    def violate_KKT(self, i, Ei):
        '''
        判别第i个α是否违反了KKT条件, 违反KKT时返回True, 否则False
        '''
        alpha_i = self.alpha[i]
        y_fx = self.y[i] * Ei
        return (y_fx > self.epsilon and alpha_i > 0 or
                y_fx < -self.epsilon and alpha_i < self.C)

    def select_j_by_i(self, i, Ei):
        '''
        根据给定的alpha索引i选择alpha_j, 计算并返回j和Ej
        '''
        valid_updated_idx = [idx for idx, update in enumerate(self.updated_E) if update and idx != i]
        if valid_updated_idx: # 如果有效的E不为空, 根据启发式规则选择 |Ei - Ej| 最大的作为 j
            Eks = list(map(self.compute_E, valid_updated_idx))
            j = np.argmax([abs(Ei - Ek) for Ek in Eks])
            Ej = Eks[j]
        else: # 如果不存在, 则随机选择一个不等于i的j
            j = np.random.choice([j for j in range(self.N) if j != i])
            Ej = self.compute_E(j)
        return j, Ej

    def compute_alpha_bounds(self, i, j):
        '''
        根据旧的 alpha_i 和 alpha_j 计算更新后的 alpha_i, alpha_j 的上下界
        '''
        if self.y[i] == self.y[j]:
            sum_ij = self.alpha[i] + self.alpha[j]
            low, high = max(0, sum_ij - self.C), min(self.C, sum_ij)
        else:
            sub_ji = self.alpha[j] - self.alpha[i]
            low, high = max(0, sub_ji), min(self.C, sub_ji + self.C)
        return low, high

    def compute_eta(self, i, j):
        '''
        eta = K(xi, xi) + K(xj, xj) - 2 * K(xi, xj), K 为核函数
        '''
        xi, xj = self.x[i], self.x[j]
        return self.kernel(xi, xi) + self.kernel(xj, xj) - 2 * self.kernel(xi, xj)

    def compute_alpha_j(self, j, Ei, Ej, eta):
        '''
        未裁剪的alpha_j: alpha_j_new = alpha_j_old * y_i * (Ei - Ej) / eta
        '''
        return self.alpha[j] + self.y[j] * (Ei - Ej) / eta

    def compute_alpha_i(self, i, j, old_alpha_j):
        '''
        alpha_i_new = alpha_i_old + y[i] * y[j] * (old_alpha_j - new_alpha_j)
        '''
        return self.alpha[i] + self.y[i] * self.y[j] * (old_alpha_j - self.alpha[j])

    def clip_alpha(self, alpha, low, high):
        '''
        裁剪α参数, 使得 low <= α <= high
        '''
        return min(high, max(low, alpha))

    def compute_b(self, i, j, Ei, Ej, alpha_i_old, alpha_j_old):
        '''
        更新b, b的更新规则与 alpha_i_new 和 alpha_j_new 的范围有关
        '''
        xi, xj = self.x[i], self.x[j]
        yi, yj = self.y[i], self.y[j]
        alpha_i, alpha_j = self.alpha[i], self.alpha[j] # new_alpha_i 以及 new_alpha_j

        bi = -Ei - yi * self.kernel(xi, xi) * (alpha_i - alpha_i_old)\
             -yj * self.kernel(xj, xi) * (alpha_j - alpha_j_old) + self.b
        bj = -Ej - yi * self.kernel(xi, xj) * (alpha_i- alpha_i_old)\
             -yj * self.kernel(xj, xj) * (alpha_j - alpha_j_old) + self.b
        if 0 < alpha_i < self.C:
            return bi
        elif 0 < alpha_j < self.C:
            return bj
        else:
            return (bi + bj) / 2