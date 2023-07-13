import numpy as np
from matplotlib import pyplot as plt
from typing import Union, Callable, Sequence, Annotated, Literal, Optional
from numbers import Number
from numpy.typing import ArrayLike, NDArray
from numpy.linalg import inv, pinv
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import lsqr, spsolve
from scipy.linalg import solve_banded, solveh_banded
from .boundary_condition import BoundaryCondition, DirichletBoundaryCondition
from matplotlib.tri import Triangulation
from .matrix import kronecker_delta


class FEMSolver1D(object):
    def __init__(self,
                 alpha_func: Union[Callable[[float], float], float],
                 beta_func: Union[Callable[[float], float], float],
                 f_func: Union[Callable[[float], float], float],
                 boundary_conditions: Sequence[BoundaryCondition]):
        self.alpha_func = alpha_func
        self.beta_func = beta_func
        self.f_func = f_func
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
        
        if isinstance(self.f_func, Callable):
            f = self.f_func(mid_x_arr)
        else:
            f = np.ones(segments_num) * self.f_func
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
        for i in range(1, np.shape(K_matrix)[0] - 1):
            K_matrix[i, i] = K_22[i - 1] + K_11[i]
            # K_matrix[i, i-1] = K_21[i-1]
            # K_matrix[i-1, i] = K_12[i-1]
            K_matrix[i + 1, i] = K_21[i]
            K_matrix[i, i + 1] = K_12[i]
        
        # 常数矩阵 b
        b_matrix = np.zeros(segments_num + 1)
        b_matrix[0] = f[0] * ell[0] / 2
        b_matrix[-1] = f[-1] * ell[-1] / 2
        for i in range(1, len(b_matrix) - 1):
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
                 alpha_x_func: Union[Callable[[float, float], float], float],
                 alpha_y_func: Union[Callable[[float, float], float], float],
                 beta_func: Union[Callable[[float, float], float], float],
                 f_func: Union[Callable[[float, float], float], float],
                 boundary_conditions: Sequence[BoundaryCondition],
                 *,
                 triangulation: Optional[Triangulation] = None):
        self.alpha_x_func = alpha_x_func
        self.alpha_y_func = alpha_y_func
        self.beta_func = beta_func
        self.f_func = f_func
        
        
        
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
            
            
        # 中间变量
        self._triangulation = triangulation
        self.ns_mat = None
        self.x_arr = None
        self.y_arr = None
        self.x_arr_3xN = None
        self.y_arr_3xN = None
        self.alpha_x = None
        self.alpha_y = None
        self.beta = None
        self.f = None
        self.a_arr_3xN = None
        self.b_arr_3xN = None
        self.c_arr_3xN = None
        self.Delta_arr = None
        self.K_arr_3x3xN = None
        self.K_mat: Optional[np.ndarray] = None
        self.b_mat_arr_3xN = None
        self.b_mat = None
        
    @property
    def triangulation(self):
        return self._triangulation
    @triangulation.setter
    def triangulation(self, input_tri: Triangulation):
        self._triangulation = input_tri
        self.ns_mat = self._triangulation.triangles
        self.x_arr = self._triangulation.x
        self.y_arr = self._triangulation.y
        self.x_arr_3xN, self.y_arr_3xN = self._cal_xy_arr_3xN()
        self.alpha_x, self.alpha_y, self.beta, self.f = self._cal_param_arr()
        self.a_arr_3xN, self.b_arr_3xN, self.c_arr_3xN = self._cal_abc_arr_3xN()
        self.Delta_arr = self._cal_Delta_arr()
        self.K_arr_3x3xN = self._cal_K_arr_3x3xN()
        self.K_mat = self._cal_K_mat()
        self.b_mat_arr_3xN = self._cal_b_mat_arr_3xN()
        self.b_mat = self._cal_b_mat()
    
    def _cal_param_arr(self) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        
        area_num = np.shape(self.ns_mat)[0]
        # alpha, beta, f 矩阵,
        func_lst = [self.alpha_x_func, self.alpha_y_func, self.beta_func, self.f_func]
        param_lst = []
        # 如果给的是函数, 则计算各个矩阵; 如果给的是常数, 则赋予常数阵
        for i in range(len(func_lst)):
            if isinstance(func_lst[i], Callable):
                # FIXME: 应该使用每个单元的值, 而不是节点处的值
                param_lst.append(func_lst[i](self.x_arr_3xN, self.y_arr_3xN))
            else:
                param_lst.append(np.ones(area_num) * func_lst[i])
        alpha_x, alpha_y, beta, f = param_lst
        
        return alpha_x, alpha_y, beta, f
    
    
    
    def solve(self, triangulation: Triangulation = None):
        if triangulation is not None:
            self.triangulation = triangulation
        assert self.triangulation is not None, "请指定三角形网格."
        
        Phi = np.linalg.solve(self.K_mat, self.b_mat)
        Phi = (Phi - np.min(Phi)) / (np.max(Phi)-np.min(Phi))
        
        return Phi


        
    def tripcolor(self, *, show_mesh=True):
        Phi = self.solve()
        x_arr = self.triangulation.x
        y_arr = self.triangulation.y
        triangles = self.triangulation.triangles
        
        fig, ax = plt.subplots()
        ax.tripcolor(x_arr, y_arr, Phi, triangles=triangles, linewidth=0, rasterized=True)
        
        # 画网格
        if show_mesh:
            ax.triplot(self.triangulation, color="k", lw=.5, alpha=.5)
        
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax
    
    def _cal_xy_arr_3xN(self):
        x_arr_3xN = np.zeros(np.shape(self.ns_mat.transpose()))
        y_arr_3xN = np.zeros(np.shape(self.ns_mat.transpose()))
        
        x_arr_3xN[0] = self.x_arr[self.ns_mat[:, 0]]
        x_arr_3xN[1] = self.x_arr[self.ns_mat[:, 1]]
        x_arr_3xN[2] = self.x_arr[self.ns_mat[:, 2]]
        
        y_arr_3xN[0] = self.y_arr[self.ns_mat[:, 0]]
        y_arr_3xN[1] = self.y_arr[self.ns_mat[:, 1]]
        y_arr_3xN[2] = self.y_arr[self.ns_mat[:, 2]]
        
        return x_arr_3xN, y_arr_3xN
    def _cal_Delta_arr(self) -> ArrayLike:
        Delta_arr = (self.b_arr_3xN[0] * self.c_arr_3xN[1] - self.b_arr_3xN[1] * self.c_arr_3xN[0]) / 2
        return Delta_arr
    
    def _cal_N_arr(self) -> ArrayLike:
        # N_arr = kronecker_delta()
        pass
    
    def _cal_K_arr_3x3xN(self) -> ArrayLike:
        arr_len = len(self.Delta_arr)
        K_arr_3x3xN = np.zeros((3, 3, arr_len))
        
        for i in range(3):
            for j in range(3):
                tmp1 = self.alpha_x * self.b_arr_3xN[i] * self.b_arr_3xN[j] + self.alpha_y * self.c_arr_3xN[i] * self.c_arr_3xN[j]
                tmp2 = self.beta * (1 + kronecker_delta(i, j))
                K_arr_3x3xN[i, j] = tmp1 / (4 * self.Delta_arr) + self.Delta_arr / 12 * tmp2
                
        return K_arr_3x3xN
    
    def _cal_abc_arr_3xN(self):
        arr_len = len(self.ns_mat)
        a_arr_3xN = np.zeros((3, arr_len))
        b_arr_3xN = np.zeros((3, arr_len))
        c_arr_3xN = np.zeros((3, arr_len))
        
        a_arr_3xN[0] = self.x_arr_3xN[1] * self.y_arr_3xN[2] - self.y_arr_3xN[1] * self.x_arr_3xN[2]
        a_arr_3xN[1] = self.x_arr_3xN[2] * self.y_arr_3xN[0] - self.y_arr_3xN[2] * self.x_arr_3xN[0]
        a_arr_3xN[2] = self.x_arr_3xN[0] * self.y_arr_3xN[1] - self.y_arr_3xN[0] * self.x_arr_3xN[1]
        
        b_arr_3xN[0] = self.y_arr_3xN[1] - self.y_arr_3xN[2]
        b_arr_3xN[1] = self.y_arr_3xN[2] - self.y_arr_3xN[0]
        b_arr_3xN[2] = self.y_arr_3xN[0] - self.y_arr_3xN[1]
        
        c_arr_3xN[0] = self.x_arr_3xN[2] - self.x_arr_3xN[1]
        c_arr_3xN[1] = self.x_arr_3xN[0] - self.x_arr_3xN[2]
        c_arr_3xN[2] = self.x_arr_3xN[1] - self.x_arr_3xN[0]
        
        return a_arr_3xN, b_arr_3xN, c_arr_3xN
    
    def _cal_K_mat(self) -> np.ndarray:
        dimension = np.max(self.ns_mat.flatten()) + 1
        K_mat = np.zeros((dimension, dimension))
        for e in range(np.shape(self.ns_mat)[0]):
            for i in range(3):
                row = self.ns_mat[e, i]
                for j in range(3):
                    col = self.ns_mat[e, j]
                    K_mat[row, col] += self.K_arr_3x3xN[i, j, e]
        return K_mat
    
    def _cal_b_mat(self):
        dimension = np.max(self.ns_mat.flatten()) + 1
        b_mat = np.zeros(dimension)
        for e in range(np.shape(self.ns_mat)[0]):
            for i in range(3):
                idx = self.ns_mat[e, i]
                b_mat[idx] += self.b_mat_arr_3xN[i, e]
        return b_mat
    
    def _cal_b_mat_arr_3xN(self):
        b_mat_arr_3xN = np.zeros((3, len(self.Delta_arr)))
        for e in range(len(self.Delta_arr)):
            for i in range(3):
                b_mat_arr_3xN[i, e] = self.Delta_arr[e] * self.f[e] / 3
        return b_mat_arr_3xN
    
    def plot_mesh(self, triangulation: Triangulation = None, *, show_tag=False):
        if triangulation is None:
            triangulation = self.triangulation
        
        fig, ax = plt.subplots()
        ax.triplot(triangulation, color="k", lw=.5)
        # plt.text(triangulation.x[triangulation.triangles[0]], triangulation.y[triangulation.triangles[0]], "a")
        if show_tag and np.shape(triangulation.triangles)[0] < 99:  # 如果网格过多, 也会强制不标tag
            for row_i, row in enumerate(triangulation.triangles):
                mid_x = np.mean(triangulation.x[row])
                mid_y = np.mean(triangulation.y[row])
                ax.text(mid_x, mid_y, row_i, color="r", ha="center", va="center")
                for col in row:
                    # TODO 优化 text, 单次text
                    ax.text(triangulation.x[col],
                            triangulation.y[col],
                            col,
                            ha="center",
                            va="center",
                            backgroundcolor="r",
                            color="w",
                            bbox=dict(boxstyle="circle"))
                
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.set_box_aspect(1)
        ax.set_title("Triangular Mesh")
        return fig, ax

    def plot_K_mat(self):  # TODO: 增加 **kwarg, 以自定义cmap等
        tmp1 = np.max(self.K_mat.flatten())
        tmp2 = np.min(self.K_mat.flatten())
        tmp = max(tmp1, -tmp2)
        fig, ax = plt.subplots()
        im = ax.matshow(self.K_mat, cmap="seismic", vmin=-tmp, vmax=tmp)
        cb = fig.colorbar(im, ax=ax)
        
        ax.set_title("$K$ Matrix")
        
        return fig, ax
        
        
        