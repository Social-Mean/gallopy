import numpy as np
from matplotlib import pyplot as plt
from typing import Union, Callable, Sequence
from numbers import Number
from numpy.typing import ArrayLike
from numpy.linalg import inv, pinv
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import lsqr, spsolve
from scipy.linalg import solve_banded, solveh_banded
from .boundary_condition import BoundaryCondition, DirichletBoundaryCondition
from matplotlib.tri import Triangulation

class FEMSolver1D(object):
    def __init__(self,
                 alpha_func: Union[Callable, Number],
                 beta_func: Union[Callable, Number],
                 force_func: Union[Callable, Number],
                 boundary_conditions: Sequence[BoundaryCondition]):
        self.alpha_func = alpha_func
        self.beta_func = beta_func
        self.force_func = force_func
        self.boundary_conditions = boundary_conditions
        self.dirichlet_boundary_condition_list = []
        self.neumann_boundary_condition_list = []
        self.third_boundary_condition_list = []
        for condition in self.boundary_conditions:
            if isinstance(condition, DirichletBoundaryCondition):
                self.dirichlet_boundary_condition_list.append(condition)
            else:
                pass
            
    def _cal_param_matrix(self, x_array: ArrayLike) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        mid_x_arr = (x_array[:-1] + x_array[1:]) / 2
        segments_num = len(mid_x_arr)
        # alpha, beta, f 矩阵,
        # 如果给的是函数, 则计算各个矩阵; 如果给的是常数, 则赋予常数阵
        if isinstance(self.alpha_func, Callable):
            alpha = self.alpha_func(mid_x_arr)
        else:
            alpha = np.ones(segments_num) * self.alpha_func
        
        if isinstance(self.beta_func, Callable):
            beta = self.beta_func(mid_x_arr)
        else:
            beta = np.ones(segments_num) * self.beta_func
        
        if isinstance(self.force_func, Callable):
            f = self.force_func(mid_x_arr)
        else:
            f = np.ones(segments_num) * self.force_func
        return alpha, beta, f
    
    def solve(self, x_array: ArrayLike) -> ArrayLike:
        
        segments_num = len(x_array) - 1
        # 使用线段的中点值计算 alpha, beta 和 f
        
        alpha, beta, f = self._cal_param_matrix(x_array)
        
        # l 矩阵
        ell = np.zeros(segments_num)
        for i in range(segments_num):
            ell[i] = x_array[i + 1] - x_array[i]
        
        # b 向量
        b1 = f * ell / 2
        b2 = f * ell / 2
        
        # 计算 K_ii
        K_11 = alpha / ell + beta * ell / 3
        K_12 = -alpha / ell + beta * ell / 6
        K_21 = K_12
        K_22 = K_11
        
        # 系数矩阵 K
        K_matrix = np.zeros((segments_num + 1, segments_num + 1))
        K_matrix[0, 0] = K_11[0]
        K_matrix[-1, -1] = K_22[-1]
        for i in range(1, np.shape(K_matrix)[0]-1):
            K_matrix[i, i] = K_22[i-1] + K_11[i]
            # K_matrix[i, i-1] = K_21[i-1]
            # K_matrix[i-1, i] = K_12[i-1]
            K_matrix[i+1, i] = K_21[i]
            K_matrix[i, i+1] = K_12[i]
        
        # 常数矩阵 b
        b_matrix = np.zeros(segments_num + 1)
        b_matrix[0] = f[0] * ell[0] / 2
        b_matrix[-1] = f[-1] * ell[-1] / 2
        for i in range(1, len(b_matrix)-1):
            b_matrix[i] = b2[i - 1] + b1[i]
        
        # 施加狄利克雷边界条件
        for condition in self.dirichlet_boundary_condition_list:
            # 找到边界条件对应的索引
            index = np.argwhere(x_array == condition.x_val)
            
            # 如果所给的边界条件的 x 值不在 x_array 中, 则将这个 x 插入到 x_array 中
            if np.size(index) == 0:
                x_array = np.append(x_array, condition.x_val)
                x_array = np.sort(x_array)
                index = np.argwhere(x_array == condition.x_val)
                
            K_matrix[index] = np.zeros(np.shape(K_matrix)[1])
            K_matrix[index, index] = 1
            # b_matrix -= K_matrix[:, 0] * condition.y_val
            b_matrix[index] = condition.y_val
        # TODO: 使用稀疏矩阵提高效率
        y_array = pinv(K_matrix) @ b_matrix
        # y_array = lsqr(K_matrix, b_matrix)[0]
        
        return y_array

# class LaplaceSolver1D(FEMSolver1D):
#     def __init__(self):


class FEMSolver2D(object):
    def __init__(self,
                 alpha_x_func: Union[Callable, Number],
                 alpha_y_func: Union[Callable, Number],
                 beta_func: Union[Callable, Number],
                 force_func: Union[Callable, Number],
                 boundary_conditions: Sequence[BoundaryCondition]):
        self.alpha_x_func = alpha_x_func
        self.alpha_y_func = alpha_y_func
        self.beta_func = beta_func
        self.force_func = force_func
        self.boundary_conditions = boundary_conditions
        
        self.dirichlet_boundary_condition_list = []
        self.neumann_boundary_condition_list = []
        self.third_boundary_condition_list = []
        
        # 边界条件的分类
        for condition in self.boundary_conditions:
            if isinstance(condition, DirichletBoundaryCondition):
                self.dirichlet_boundary_condition_list.append(condition)
            else:
                pass
    
    def _cal_param_matrix(self, X_arr, Y_arr) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        # segments_num = np.shape(X_arr) - np.array([1, 1])
        
        mid_X_arr = (X_arr[:-1, :-1] + X_arr[1:, 1:]) / 2
        mid_Y_arr = (Y_arr[:-1, :-1] + Y_arr[1:, 1:]) / 2
        
        segments_num = np.shape(mid_X_arr)
        # alpha, beta, f 矩阵,
        # 如果给的是函数, 则计算各个矩阵; 如果给的是常数, 则赋予常数阵
        if isinstance(self.alpha_x_func, Callable):
            alpha_x = self.alpha_x_func(mid_X_arr, mid_Y_arr)
        else:
            alpha_x = np.ones(segments_num) * self.alpha_x_func
        
        if isinstance(self.alpha_y_func, Callable):
            alpha_y = self.alpha_y_func(mid_X_arr, mid_Y_arr)
        else:
            alpha_y = np.ones(segments_num) * self.alpha_y_func
        
        if isinstance(self.beta_func, Callable):
            beta = self.beta_func(mid_X_arr, mid_Y_arr)
        else:
            beta = np.ones(segments_num) * self.beta_func
        
        if isinstance(self.force_func, Callable):
            f = self.force_func(mid_X_arr, mid_Y_arr)
        else:
            f = np.ones(segments_num) * self.force_func
        return alpha_x, alpha_y, beta, f
    
    def solve(self, triangulation: Triangulation):
        x_arr = triangulation.x
        y_arr = triangulation.y
        
        X_arr, Y_arr = np.meshgrid(x_arr, y_arr)
        alpha_x, alpha_y, beta, f = self._cal_param_matrix(X_arr, Y_arr)
        
        ns_mat = triangulation.triangles
        pass
    
    