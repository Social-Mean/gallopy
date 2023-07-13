import numpy as np
from matplotlib import pyplot as plt
from typing import Union, Callable, Sequence, Annotated, Literal, Optional
from numbers import Number
from numpy.typing import ArrayLike, NDArray
from numpy.linalg import inv, pinv
from scipy.sparse import dia_matrix, coo_matrix, lil_array
from scipy.sparse.linalg import lsqr, spsolve
from scipy.linalg import solve_banded, solveh_banded
from .boundary_condition import BoundaryCondition, DirichletBoundaryCondition
from matplotlib.tri import Triangulation
from .matrix import kronecker_delta
import cmcrameri.cm as cmc

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


def _find_same_element(A, B):
    for a in A:
        for b in B:
            if a == b:
                return a


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
        self.N = None
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
        self.N_arr_3xN = None
        self.Phi = None
    
    def __call__(self, triangulation: Triangulation):
        return self.solve(triangulation)
        
        
    @property
    def triangulation(self):
        return self._triangulation
    @triangulation.setter
    def triangulation(self, input_tri: Triangulation):
        self._triangulation = input_tri
        self.ns_mat = self._triangulation.triangles
        self.N = np.shape(self.ns_mat)[0]
        self.x_arr = self._triangulation.x
        self.y_arr = self._triangulation.y
        self.x_arr_3xN, self.y_arr_3xN = self._cal_xy_arr_3xN()
        self.a_arr_3xN, self.b_arr_3xN, self.c_arr_3xN = self._cal_abc_arr_3xN()
        self.Delta_arr = self._cal_Delta_arr()
        self.N_arr_3xN = self._cal_N_arr_3xN()
        self.alpha_x, self.alpha_y, self.beta, self.f = self._cal_param_arr()
        self.K_arr_3x3xN = self._cal_K_arr_3x3xN()
        self.K_mat = self._cal_K_mat()
        self.b_mat_arr_3xN = self._cal_b_mat_arr_3xN()
        self.b_mat = self._cal_b_mat()
        self._impose_boundary_conditions()

    
    def _cal_param_arr(self) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        
        area_num = np.shape(self.ns_mat)[0]
        # alpha, beta, f 矩阵,
        func_lst = [self.alpha_x_func, self.alpha_y_func, self.beta_func, self.f_func]
        param_lst = []
        # 如果给的是函数, 则计算各个矩阵; 如果给的是常数, 则赋予常数阵
        for i in range(len(func_lst)):
            if isinstance(func_lst[i], Callable):
                # FIXME: 应该使用每个单元的值, 而不是节点处的值
                # param_lst.append(func_lst[i](self.x_arr_3xN, self.y_arr_3xN))
                # 尝试使用三个节点的均值作为单元的值
                # TODO: 验证使用均值的正确性
                # Phi_arr_3xN = func_lst[i](self.x_arr_3xN, self.y_arr_3xN)
                # Phi_arr_mean_1xN = np.mean(Phi_arr_3xN, axis=0)
                # param_lst.append(Phi_arr_mean_1xN)
                
                
                # 尝试使用差值函数
                Phi_1xN = np.zeros(self.N)
                for e in range(self.N):
                    Phi_e_i = np.zeros(3)
                    for j in range(3):
                        N_j_e = self.N_arr_3xN[j][e](self.x_arr_3xN[j, e], self.y_arr_3xN[j, e])
                        Phi_j_e = func_lst[i](self.x_arr_3xN[j, e], self.y_arr_3xN[j, e])
                        Phi_e_i[j] = N_j_e * Phi_j_e
                    Phi_1xN[e] = np.sum(Phi_e_i)
                param_lst.append(Phi_1xN)
            else:
                param_lst.append(np.ones(area_num) * func_lst[i])
        alpha_x, alpha_y, beta, f = param_lst
        
        return alpha_x, alpha_y, beta, f
    
    def _impose_boundary_conditions(self):
        finder = self.triangulation.get_trifinder()
        num = 50
        # x1 = 0 * np.ones(num)
        # x2 = 1 * np.ones(num)
        # xs = np.linspace(0, 1, num)
        #
        # y1 = 0 * np.ones(num)
        # y2 = 1 * np.ones(num)
        # ys = np.linspace(0, 1, num)
        # es = []
        # es+=(list(finder(x1, ys)))
        # es+=(list(finder(x2, ys)))
        # es+=(list(finder(xs, y1)))
        # es+=(list(finder(xs, y2)))
        # print(es)
        size = np.shape(self.K_mat)[0]
        # for e in es:
        #     for node_tag in self.ns_mat[e]:
        #         self.K_mat[node_tag] = np.zeros(size)
        #         self.K_mat[node_tag, node_tag] = 1
        #         self.b_mat[node_tag] = 0
        # edges = self.triangulation.edges
        # for edge in edges:
        #     idx1, idx2 = edge
        #     if (self.x_arr[idx1] in [0, 1] and self.x_arr[idx2] in [0, 1]) or (self.y_arr[idx1] in [0, 1] and self.y_arr[idx2] in [0, 1]):
        #         self.K_mat[idx1] = np.zeros(size)
        #         self.K_mat[idx1, idx1] = 1
        #         self.b_mat[idx1] = 0
        #
        #         self.K_mat[idx2] = np.zeros(size)
        #         self.K_mat[idx2, idx2] = 1
        #         self.b_mat[idx2] = 0
        
        # 边缘三角形的邻居共同包含的节点是非边界节点
        boundary_tri_idx = []
        boundary_node_tag = []
        for i, lst in enumerate(self.triangulation.neighbors):
            if -1 in lst:
                boundary_tri_idx.append(i)
                neighbor_tag_lst = []
                for tri_tag in lst:
                    if tri_tag != -1:
                        neighbor_tag_lst.append(tri_tag)
                # for tri_tag in neighbor_tag_lst:
                if len(neighbor_tag_lst) == 1:
                    A_lst = self.triangulation.triangles[neighbor_tag_lst[0]]
                    for idx in self.triangulation.triangles[i]:
                        if idx not in A_lst:
                            boundary_node_tag.append(idx)
                    continue
                A_lst = self.triangulation.triangles[neighbor_tag_lst[0]]
                B_lst = self.triangulation.triangles[neighbor_tag_lst[1]]
                
                node_tag = _find_same_element(A_lst, B_lst)
                if node_tag is not None:
                    for idx in self.triangulation.triangles[i]:
                        if idx != node_tag:
                            boundary_node_tag.append(idx)
        
        for idx in boundary_node_tag:
            self.K_mat[idx] = np.zeros(size)
            self.K_mat[idx, idx] = 1
            if self.y_arr[idx] == 0 and self.x_arr[idx] not in [0, 1]:
                self.b_mat[idx] = 0.2
            elif self.y_arr[idx] == 1 and self.x_arr[idx] not in [0, 1]:
                self.b_mat[idx] = 0.4
            elif self.x_arr[idx] == 0:
                self.b_mat[idx] = 0.5
            else:
                self.b_mat[idx] = 0.3
        
        pass
    
    def solve(self, triangulation: Triangulation = None):
        if triangulation is not None:
            self.triangulation = triangulation
        assert self.triangulation is not None, "请指定三角形网格."
        K_csr = self.K_mat.tocsr()
        # Phi = np.linalg.solve(self.K_mat, self.b_mat)
        # Phi = (Phi - np.min(Phi)) / (np.max(Phi)-np.min(Phi))
        self.Phi = spsolve(K_csr, self.b_mat)
        
        return self.Phi


        
    def tripcolor(self, *, show_mesh=True):
        if self.Phi is None:
            self.Phi = self.solve()
        x_arr = self.triangulation.x
        y_arr = self.triangulation.y
        triangles = self.triangulation.triangles
        
        fig, ax = plt.subplots()
        vmin = np.min(self.Phi)
        vmax = np.max(self.Phi)
        im = ax.tripcolor(x_arr, y_arr, self.Phi, triangles=triangles, linewidth=0, rasterized=True, vmin=vmin, vmax=vmax)
        
        # 画网格
        if show_mesh:
            ax.triplot(self.triangulation, color="k", lw=.5, alpha=.5)
        
        # colorbar
        cb = plt.colorbar(im, ax=ax)
        # cb.set_ticks(list(cb.get_ticks()) + [vmin, vmax])

        
        ax.set_title(r"$𝛷(x, y)=2\pi^2\sin\pi x\sin\pi y$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        
        
        
        ax.set_xlim((self.x_arr.min(), self.x_arr.max()))
        ax.set_ylim((self.y_arr.min(), self.y_arr.max()))
        ax.set_box_aspect(1)
        # ax.set_xticks([])
        # ax.set_yticks([])
        return fig, ax
    
    def trisurface(self, *, show_mesh=True):
        if self.Phi is None:
            self.Phi = self.solve()
        x_arr = self.triangulation.x
        y_arr = self.triangulation.y
        triangles = self.triangulation.triangles
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set_proj_type("ortho")
        # ax.set_zlim3d(zmin=0)
        vmin = np.min(self.Phi)
        vmax = np.max(self.Phi)
        im = ax.plot_trisurf(x_arr, y_arr, self.Phi, triangles=triangles, linewidth=0, rasterized=True, vmin=vmin, vmax=vmax, cmap="viridis")
        
        # 画网格
        # if show_mesh:
        #     ax.triplot(self.triangulation, color="k", lw=.5, alpha=.5)
        
        # colorbar
        cb = plt.colorbar(im, ax=ax)
        # cb.set_ticks(list(cb.get_ticks()) + [vmin, vmax])
        
        ax.set_title(r"$𝛷(x, y)=2\pi^2\sin\pi x\sin\pi y$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        
        ax.set_xlim((self.x_arr.min(), self.x_arr.max()))
        ax.set_ylim((self.y_arr.min(), self.y_arr.max()))
        # ax.set_box_aspect(1)
        # ax.set_xticks([])
        # ax.set_yticks([])
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
    
    def _cal_N_arr_3xN(self) -> ArrayLike:
        # N_arr = kronecker_delta()
        N_arr_3xN = [[None for col in range(self.N)] for row in range(3)]
        for j in range(3):
            for e in range(self.N):
                N_arr_3xN[j][e] = lambda x, y: (self.a_arr_3xN[j, e] + self.b_arr_3xN[j, e]*x + self.c_arr_3xN[j, e]*y) / (2*self.Delta_arr[e])
        
        return N_arr_3xN
    
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
    
    def _cal_K_mat(self) -> lil_array:
        dimension = np.max(self.ns_mat.flatten()) + 1
        # K_mat = np.zeros((dimension, dimension))
        K_mat = lil_array((dimension, dimension))
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
        
        ax.set_title("Triangular Mesh")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_xlim((self.x_arr.min(), self.x_arr.max()))
        ax.set_ylim((self.y_arr.min(), self.y_arr.max()))
        ax.set_box_aspect(1)
        return fig, ax

    def plot_K_mat(self):  # TODO: 增加 **kwarg, 以自定义cmap等
        # tmp1 =
        # tmp2 = np.min(self.K_mat.flatten())
        K_mat_arr = self.K_mat.toarray()
        max_abs_K = np.max(np.abs(K_mat_arr.flatten()))
        fig, ax = plt.subplots()
        im = ax.matshow(K_mat_arr/max_abs_K, cmap="seismic", vmin=-1, vmax=1)
        cb = fig.colorbar(im, ax=ax)
        
        tmp_len = np.shape(K_mat_arr)[0]
        # ax.set_xlim((0, tmp_len))
        # ax.set_ylim((0, tmp_len))
        ax.tick_params(axis="both", direction="out")
        ax.set_title(r"$K/\max|K|$ Matrix")
        
        return fig, ax
        
        
        