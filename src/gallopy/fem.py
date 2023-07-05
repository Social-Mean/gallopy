import numpy as np
from matplotlib import pyplot as plt
from typing import Union, Callable
from numbers import Number
from numpy.typing import ArrayLike
from numpy.linalg import inv
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import lsqr, spsolve
from scipy.linalg import solve_banded, solveh_banded


class FEMSolver1D(object):
    def __init__(self,
                 alpha_func: Union[Callable, Number],
                 beta_func: Union[Callable, Number],
                 force_func: Union[Callable, Number]):
        self.alpha_func = alpha_func
        self.beta_func = beta_func
        self.force_func = force_func

    def solve(self,
              x_array: ArrayLike) -> ArrayLike:
        segments_num = len(x_array) - 1
        
        # alpha, beta, f 矩阵,
        # 如果给的是函数, 则计算各个矩阵; 如果给的是常数, 则赋予常数阵
        if isinstance(self.alpha_func, Callable):
            alpha = self.alpha_func(x_array[:-1])
        else:
            alpha = np.ones(segments_num) * self.alpha_func
            
        if isinstance(self.beta_func, Callable):
            beta = self.beta_func(x_array[:-1])
        else:
            beta = np.ones(segments_num) * self.beta_func
            
        if isinstance(self.force_func, Callable):
            f = self.force_func(x_array[:-1])
        else:
            f = np.ones(segments_num) * self.force_func
        
        # l 矩阵
        ell = np.zeros(segments_num)
        for i in range(segments_num):
            ell[i] = x_array[i+1] - x_array[i]
            
        # b 向量
        b1 = f * ell / 2
        b2 = f * ell / 2
            
        # 计算 K_ii
        K_11 = alpha/ell + beta*ell/3
        K_12 = -alpha/ell + beta*ell/6
        K_21 = K_12
        K_22 = K_11
        
        # 系数矩阵 K
        K_matrix = np.zeros((segments_num+1, segments_num+1))
        for i in range(segments_num):
            K_i = np.zeros((segments_num+1, segments_num+1))
            K_i[i, i] = K_11[i]
            K_i[i, i+1] = K_12[i]
            K_i[i+1, i] = K_21[i]
            K_i[i+1, i+1] = K_22[i]
            
            K_matrix += K_i
        K_matrix_sparse = dia_matrix(K_matrix)
        # 常数矩阵 b
        b_matrix = np.zeros(segments_num+1)
        b_matrix[0] = f[0] * ell[0] / 2
        b_matrix[-1] = f[-1] * ell[-1] / 2
        for i in range(1, segments_num):
            b_matrix[i] = b1[i-1] + b2[i]
        
        # y_array = inv(K_matrix) @ b_matrix
        y_array = lsqr(K_matrix, b_matrix)[0]
        
        return y_array
    