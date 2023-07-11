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
from .matrix import kronecker_delta


    
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
    
    def _cal_param_arr(self,
                       ns_mat: ArrayLike,
                       x_arr: ArrayLike,
                       y_arr: ArrayLike) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        # segments_num = np.shape(X_arr) - np.array([1, 1])
        # X_arr, Y_arr = np.meshgrid(x_arr, y_arr)
        #
        # mid_X_arr = (X_arr[:-1, :-1] + X_arr[1:, 1:]) / 2
        # mid_Y_arr = (Y_arr[:-1, :-1] + Y_arr[1:, 1:]) / 2
        
        segments_num = np.shape(ns_mat)[0]
        # alpha, beta, f 矩阵,
        # 如果给的是函数, 则计算各个矩阵; 如果给的是常数, 则赋予常数阵
        # TODO alpha_x 等都是向量, 索引是e
        if isinstance(self.alpha_x_func, Callable):
            alpha_x = self.alpha_x_func(x_arr, y_arr)
        else:
            alpha_x = np.ones(segments_num) * self.alpha_x_func
            # alpha_x = self.alpha_x_func
        
        if isinstance(self.alpha_y_func, Callable):
            alpha_y = self.alpha_y_func(x_arr, y_arr)
        else:
            alpha_y = np.ones(segments_num) * self.alpha_y_func
            # alpha_y = self.alpha_y_func
        
        if isinstance(self.beta_func, Callable):
            beta = self.beta_func(x_arr, y_arr)
        else:
            beta = np.ones(segments_num) * self.beta_func
            # beta = self.beta_func
        
        if isinstance(self.force_func, Callable):
            f = self.force_func(x_arr, y_arr)
        else:
            f = np.ones(segments_num) * self.force_func
        return alpha_x, alpha_y, beta, f
    
    def solve(self, triangulation: Triangulation):
        ns_mat = triangulation.triangles

        x_arr = triangulation.x
        y_arr = triangulation.y
        
        alpha_x, alpha_y, beta, f = self._cal_param_arr(ns_mat, x_arr, y_arr)
        a_arr_3x0, b_arr_3x0, c_arr_3x0 = self._cal_abc_arr_3x0(ns_mat, x_arr, y_arr)
        Delta_arr = self._cal_Delta_arr(b_arr_3x0, c_arr_3x0)
        
        N_arr = self._cal_N_arr(x_arr, y_arr)
        K_arr_3x3x0 = self._cal_K_arr_3x3x0(alpha_x, alpha_y, beta, Delta_arr, b_arr_3x0, c_arr_3x0)
        pass
    
    def _cal_Delta_arr(self, b_arr_3x0: ArrayLike, c_arr_3x0: ArrayLike) -> ArrayLike:
     
        
        Delta_arr = (b_arr_3x0[0]*c_arr_3x0[1] - b_arr_3x0[1]*c_arr_3x0[0]) / 2
        return Delta_arr
    
    def _cal_N_arr(self, x_arr: ArrayLike, y_arr: ArrayLike) -> ArrayLike:
        # N_arr = kronecker_delta()
        pass
    
    def _cal_K_arr_3x3x0(self,
                         alpha_x: ArrayLike,
                         alpha_y: ArrayLike,
                         beta: ArrayLike,
                         Delta_arr: ArrayLike,
                         b_arr_3x0: ArrayLike,
                         c_arr_3x0: ArrayLike) -> ArrayLike:
        arr_len = len(Delta_arr)
        K_arr_3x3x0 = np.zeros((3, 3, arr_len))
        
        for i in range(3):
            for j in range(3):
                tmp1 = alpha_x * b_arr_3x0[i] * b_arr_3x0[j] + alpha_y * c_arr_3x0[i] * c_arr_3x0[j]
                tmp2 = beta*(1+kronecker_delta(i, j))
                K_arr_3x3x0[i, j] = tmp1 / (4*Delta_arr) + Delta_arr/12 * tmp2
        
        return K_arr_3x3x0
        
    def _cal_abc_arr_3x0(self, ns_mat, x_arr, y_arr):
        arr_len = len(ns_mat)
        a_arr_3x0 = np.zeros((3, arr_len))
        b_arr_3x0 = np.zeros((3, arr_len))
        c_arr_3x0 = np.zeros((3, arr_len))
        
        ns1, ns2, ns3 = np.hsplit(ns_mat, 3)
        ns1 = ns1.flatten()
        ns2 = ns2.flatten()
        ns3 = ns3.flatten()
        
        x1_arr = x_arr[ns1]
        x2_arr = x_arr[ns2]
        x3_arr = x_arr[ns3]
        
        y1_arr = y_arr[ns1]
        y2_arr = y_arr[ns2]
        y3_arr = y_arr[ns3]
        
        a_arr_3x0[0] = x2_arr*y3_arr - y2_arr*x3_arr
        a_arr_3x0[1] = x3_arr*y1_arr - y3_arr*x1_arr
        a_arr_3x0[2] = x1_arr*y2_arr - y1_arr*x2_arr
        
        b_arr_3x0[0] = y2_arr - y3_arr
        b_arr_3x0[1] = y3_arr - y1_arr
        b_arr_3x0[2] = y1_arr - y2_arr
        
        c_arr_3x0[0] = x3_arr - x2_arr
        c_arr_3x0[1] = x1_arr - x3_arr
        c_arr_3x0[2] = x2_arr - x1_arr
        
        return a_arr_3x0, b_arr_3x0, c_arr_3x0
        
    
    