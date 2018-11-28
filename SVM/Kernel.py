# coding=utf8
import warnings
import numpy as np

class AbstractSVMKernel(object):
    '''
    核函数类的基类, 计算核函数与内积
    '''

    def __init__(self, **kernel_params):
           pass

    def __call__(self, xi, xj):
        raise NotImplementedError('Abstract inner product not implemented.')


class LinearKernel(AbstractSVMKernel):
    '''
    线性核函数类
    np.dot(xi.T, xj)
    '''

    def __call__(self, xi, xj):
        '''
        线性核内积
        np.dot(xi, xj.T)
        '''
        return np.dot(xi, xj.T)


class RBFKernel(AbstractSVMKernel):
    '''
    RBF核函数类
    exp(-gamma * L2norm(xi-xj))
    '''
    def __init__(self, **kernel_params):
        '''
        RBF核参数
        gamma:
            默认值为 (1 / k), k 为类别数, 由Kernel类计算
            缺省值 0.1
        '''
        self.gamma = kernel_params.get('gamma', 0.1)

    def __call__(self, xi, xj):
        '''
        RBF核函数内积
        np.exp(-gamma * L2norm(xi-xj))
        这里指定 axis=-1 为了兼容 xj 为训练数据矩阵和 xj 为向量2中情况
        '''
        return np.exp(-self.gamma * np.linalg.norm(xi - xj, ord=2, axis=-1))


class PolynomialKernal(AbstractSVMKernel):
    '''
    多项式核函数类
    (gamma * np.dot(xi.T, xj) + r)**d
    '''
    def __init__(self, **kernel_params):
        '''
        多项式核参数
        gamma:
            默认值 (1 / k), k 为类别数, 由Kernel类计算
            缺省值 0.1
        d: 多项式核的最高次数
            默认值 3
            缺省值 3
        r: 相关系数coefficient
            默认值 0
            缺省值 0
        '''
        self.gamma = kernel_params.get('gamma', 0.1)
        self.d = kernel_params.get('d', 3)
        self.r = kernel_params.get('r', 0)

    def __call__(self, xi, xj):
        '''
        多项式核内积
        (gamma * np.dot(xi.T, xj) + r)**d
        '''
        return (self.gamma * np.dot(xi, xj.T) + self.r)**self.d


class SigmoidKernel(AbstractSVMKernel):
    '''
    Sigmoid核函数
    注意: Sigmoid核采用的是tanh函数
    tanh(gamma * np.dot(xi, xj.T) + r)
    '''
    def __init__(self, **kernel_params):
        '''
        Sigmoid核参数
        gamma:
            默认值 (1 / k), k 为类别数, 由Kernel类计算
            缺省值 0.1
        r: 相关系数coefficient
            默认值 0
            缺省值 0
        '''
        self.gamma = kernel_params.get('gamma', 0.1)
        self.r = kernel_params.get('r', 0)
    
    def __call__(self, xi, xj):
        '''
        Sigmoid核内积
        tanh(gamma * np.dot(xi.T, xj) + r)
        '''
        return np.tanh(self.gamma * np.dot(xi, xj.T) + self.r)


def kernel(kernel_name=None, **kernel_params):
    '''
    根据指定参数返回核函数实例
    '''
    name2kernel = {
        None: LinearKernel,
        'linear': LinearKernel,
        'rbf': RBFKernel,
        'poly': PolynomialKernal,
        'sigmoid':SigmoidKernel,
    }
    assert kernel_name is None or \
        (kernel_name.lower() in name2kernel.keys()),\
        f"invalid kernel: '{str(kernel_name)}'." + "\n" \
        "The following kernels are supported:\n" + f"{list(name2kernel.keys())}"
    
    if ('gamma' not in kernel_params
            and kernel_name in ['rbf', 'poly', 'sigmoid']):
        if 'y' in kernel_params:
            kernel_params['gamma'] = 1 / len(set(kernel_params['y']))
            kernel_params.pop('y')
        else:
            warnings.warn("No label or gamma found, using default value for gamma...")
    return name2kernel.get(kernel_name, LinearKernel)(**kernel_params)
