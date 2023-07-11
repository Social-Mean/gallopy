import numpy as np
from matplotlib import pyplot as plt
from typing import Union, Callable, Sequence, Annotated, Literal
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
                 boundary_conditions: Sequence[BoundaryCondition]):
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
    
    def _cal_param_arr(self,
                       ns_mat: ArrayLike,
                       x_arr_3xN: ArrayLike,
                       y_arr_3xN: ArrayLike) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        
        area_num = np.shape(ns_mat)[0]
        # alpha, beta, f 矩阵,
        func_lst = [self.alpha_x_func, self.alpha_y_func, self.beta_func, self.f_func]
        param_lst = []
        # 如果给的是函数, 则计算各个矩阵; 如果给的是常数, 则赋予常数阵
        for i in range(len(func_lst)):
            if isinstance(func_lst[i], Callable):
                param_lst.append(func_lst[i](x_arr_3xN, y_arr_3xN))
            else:
                param_lst.append(np.ones(area_num) * func_lst[i])
        alpha_x, alpha_y, beta, f = param_lst
        
        return alpha_x, alpha_y, beta, f
    
    def solve(self, triangulation: Triangulation):
        ns_mat = triangulation.triangles
        
        x_arr = triangulation.x
        y_arr = triangulation.y
        
        x_arr_3xN, y_arr_3xN = self._cal_xy_arr_3xN(ns_mat, x_arr, y_arr)
        
        alpha_x, alpha_y, beta, f = self._cal_param_arr(ns_mat, x_arr_3xN, y_arr_3xN)
        a_arr_3xN, b_arr_3xN, c_arr_3xN = self._cal_abc_arr_3xN(ns_mat, x_arr_3xN, y_arr_3xN)
        Delta_arr = self._cal_Delta_arr(b_arr_3xN, c_arr_3xN)
        
        N_arr = self._cal_N_arr(x_arr_3xN, y_arr_3xN)
        K_arr_3x3xN = self._cal_K_arr_3x3xN(alpha_x, alpha_y, beta, Delta_arr, b_arr_3xN, c_arr_3xN)
        
        K_mat = self._cal_K_mat(ns_mat, K_arr_3x3xN)
        # plt.figure()
        # tmp1 = np.max(K_mat.flatten())
        # tmp2 = np.min(K_mat.flatten())
        # tmp = max(tmp1, -tmp2)
        # plt.matshow(K_mat, cmap="seismic", vmin=-tmp, vmax=tmp)
        # # plt.show()
        # plt.savefig("./outputs/K_mat.pdf")
        b_mat_arr_3xN = self._cal_b_mat_arr_3xN(Delta_arr, f)
        b_mat = self._cal_b_mat(ns_mat, b_mat_arr_3xN)
        fig, ax = plt.subplots()
        # Phi = inv(K_mat) @ b_mat
        Phi = np.linalg.solve(K_mat, b_mat)
        Phi = (Phi - np.min(Phi)) / (np.max(Phi)-np.min(Phi))
        
        X_arr, Y_arr = np.meshgrid(x_arr, y_arr)
        # ax.pcolormesh(X_arr, Y_arr, Phi)
        color = np.zeros((len(Phi), 3))
        color[:, 2] = Phi
        # ax.scatter(x_arr, y_arr, c=color)
        # ax.pcolormesh()
        ax.tripcolor(x_arr, y_arr, Phi, triangles=triangulation.triangles, linewidth=0, rasterized=True)
        # ax.triplot(x_arr, y_arr, Phi, triangles=triangulation.triangles)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig("./outputs/Phi.pdf")
        
    
    def _cal_xy_arr_3xN(self, ns_mat, x_arr, y_arr):
        x_arr_3xN = np.zeros(np.shape(ns_mat.transpose()))
        y_arr_3xN = np.zeros(np.shape(ns_mat.transpose()))
        
        x_arr_3xN[0] = x_arr[ns_mat[:, 0]]
        x_arr_3xN[1] = x_arr[ns_mat[:, 1]]
        x_arr_3xN[2] = x_arr[ns_mat[:, 2]]
        
        y_arr_3xN[0] = y_arr[ns_mat[:, 0]]
        y_arr_3xN[1] = y_arr[ns_mat[:, 1]]
        y_arr_3xN[2] = y_arr[ns_mat[:, 2]]
        
        return x_arr_3xN, y_arr_3xN
    def _cal_Delta_arr(self, b_arr_3xN: ArrayLike, c_arr_3xN: ArrayLike) -> ArrayLike:
        
        Delta_arr = (b_arr_3xN[0] * c_arr_3xN[1] - b_arr_3xN[1] * c_arr_3xN[0]) / 2
        return Delta_arr
    
    def _cal_N_arr(self, x_arr: ArrayLike, y_arr: ArrayLike) -> ArrayLike:
        # N_arr = kronecker_delta()
        pass
    
    def _cal_K_arr_3x3xN(self,
                         alpha_x: ArrayLike,
                         alpha_y: ArrayLike,
                         beta: ArrayLike,
                         Delta_arr: ArrayLike,
                         b_arr_3xN: ArrayLike,
                         c_arr_3xN: ArrayLike) -> ArrayLike:
        arr_len = len(Delta_arr)
        K_arr_3x3xN = np.zeros((3, 3, arr_len))
        
        for i in range(3):
            for j in range(3):
                tmp1 = alpha_x * b_arr_3xN[i] * b_arr_3xN[j] + alpha_y * c_arr_3xN[i] * c_arr_3xN[j]
                tmp2 = beta * (1 + kronecker_delta(i, j))
                K_arr_3x3xN[i, j] = tmp1 / (4 * Delta_arr) + Delta_arr / 12 * tmp2
        
        return K_arr_3x3xN
    
    def _cal_abc_arr_3xN(self, ns_mat, x_arr_3xN, y_arr_3xN):
        arr_len = len(ns_mat)
        a_arr_3xN = np.zeros((3, arr_len))
        b_arr_3xN = np.zeros((3, arr_len))
        c_arr_3xN = np.zeros((3, arr_len))
        
        # ns1, ns2, ns3 = np.hsplit(ns_mat, 3)
        # ns1 = ns1.flatten()
        # ns2 = ns2.flatten()
        # ns3 = ns3.flatten()
        #
        # x1_arr = x_arr[ns1]
        # x2_arr = x_arr[ns2]
        # x3_arr = x_arr[ns3]
        #
        # y1_arr = y_arr[ns1]
        # y2_arr = y_arr[ns2]
        # y3_arr = y_arr[ns3]
        
        a_arr_3xN[0] = x_arr_3xN[1] * y_arr_3xN[2] - y_arr_3xN[1] * x_arr_3xN[2]
        a_arr_3xN[1] = x_arr_3xN[2] * y_arr_3xN[0] - y_arr_3xN[2] * x_arr_3xN[0]
        a_arr_3xN[2] = x_arr_3xN[0] * y_arr_3xN[1] - y_arr_3xN[0] * x_arr_3xN[1]
        
        b_arr_3xN[0] = y_arr_3xN[1] - y_arr_3xN[2]
        b_arr_3xN[1] = y_arr_3xN[2] - y_arr_3xN[0]
        b_arr_3xN[2] = y_arr_3xN[0] - y_arr_3xN[1]
        
        c_arr_3xN[0] = x_arr_3xN[2] - x_arr_3xN[1]
        c_arr_3xN[1] = x_arr_3xN[0] - x_arr_3xN[2]
        c_arr_3xN[2] = x_arr_3xN[1] - x_arr_3xN[0]
        
        return a_arr_3xN, b_arr_3xN, c_arr_3xN
    
    def _cal_K_mat(self,
                   ns_mat: Annotated[NDArray[float], Literal["N", 3]],
                   K_arr_3x3xN: Annotated[NDArray[float], Literal[3, 3, "N"]]) -> np.ndarray:
        dimension = np.max(ns_mat.flatten()) + 1
        K_mat = np.zeros((dimension, dimension))
        for e in range(np.shape(ns_mat)[0]):
            for i in range(3):
                row = ns_mat[e, i]
                for j in range(3):
                    col = ns_mat[e, j]
                    K_mat[row, col] += K_arr_3x3xN[i, j, e]
        return K_mat
    
    def _cal_b_mat(self,
                   ns_mat,
                   b_mat_arr_3xN):
        dimension = np.max(ns_mat.flatten()) + 1
        b_mat = np.zeros(dimension)
        for e in range(np.shape(ns_mat)[0]):
            for i in range(3):
                idx = ns_mat[e, i]
                b_mat[idx] += b_mat_arr_3xN[i, e]
        return b_mat
    
    def _cal_b_mat_arr_3xN(self, Delta_arr, f):
        b_mat_arr_3xN = np.zeros((3, len(Delta_arr)))
        for e in range(len(Delta_arr)):
            for i in range(3):
                b_mat_arr_3xN[i, e] = Delta_arr[e] * f[i, e] / 3
        return b_mat_arr_3xN
    
    def plot_mesh(self, triangulation: Triangulation):
        fig, ax = plt.subplots()
        ax.triplot(triangulation, color="k")
        # plt.text(triangulation.x[triangulation.triangles[0]], triangulation.y[triangulation.triangles[0]], "a")
        for row_i, row in enumerate(triangulation.triangles):
            for col in row:
                ax.text(triangulation.x[col],
                        triangulation.y[col],
                        col,
                        ha="center",
                        va="center",
                        backgroundcolor="r",
                        color="w",
                        bbox=dict(boxstyle="circle"))
                mid_x = np.mean(triangulation.x[row])
                mid_y = np.mean(triangulation.y[row])
                ax.text(mid_x, mid_y, row_i, color="r", ha="center", va="center")
                ax.set_xlim((0, 1))
                ax.set_ylim((0, 1))
                ax.set_box_aspect(1)
                ax.set_title("Triangular Mesh")
        return fig, ax

    def plot_K_mat(self, triangulation: Triangulation):
        ns_mat = triangulation.triangles
        
        x_arr = triangulation.x
        y_arr = triangulation.y
        
        x_arr_3xN, y_arr_3xN = self._cal_xy_arr_3xN(ns_mat, x_arr, y_arr)
        
        alpha_x, alpha_y, beta, f = self._cal_param_arr(ns_mat, x_arr_3xN, y_arr_3xN)
        a_arr_3xN, b_arr_3xN, c_arr_3xN = self._cal_abc_arr_3xN(ns_mat, x_arr_3xN, y_arr_3xN)
        Delta_arr = self._cal_Delta_arr(b_arr_3xN, c_arr_3xN)
        
        K_arr_3x3xN = self._cal_K_arr_3x3xN(alpha_x, alpha_y, beta, Delta_arr, b_arr_3xN, c_arr_3xN)
        
        K_mat = self._cal_K_mat(ns_mat, K_arr_3x3xN)
        tmp1 = np.max(K_mat.flatten())
        tmp2 = np.min(K_mat.flatten())
        tmp = max(tmp1, -tmp2)
        fig, ax = plt.subplots()
        im = ax.matshow(K_mat, cmap="seismic", vmin=-tmp, vmax=tmp)
        cb = fig.colorbar(im, ax=ax)
        
        ax.set_title("$K$ Matrix")
        
        return fig, ax
        
        
        