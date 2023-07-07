from typing import Sequence, Union
from collections.abc import Iterable
import matplotlib as mpl
import numpy as np
from numpy.typing import ArrayLike
from numpy.fft import fftshift, fftn
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from . import rcParams
from .matrix import convmat
from matplotlib import patches, collections


class KeyPoint(object):
    def __init__(self, name: str, position: ArrayLike):
        self.name = name
        self.position = position


class PWEMSolver(object):
    def __init__(self, epsilon_r, mu_r, lattice_constant):
        self.epsilon_r = epsilon_r
        self.mu_r = mu_r
        self.lattice_constant = lattice_constant
    
    def solve_path(self,
                   P: int,
                   Q: int,
                   mode,
                   key_points: Sequence[KeyPoint],
                   num: Union[int, Sequence] = 50,
                   *,
                   return_num_list: bool = False):
        
        # 总的空间谐波数
        spatial_harmonic_wave_num = P * Q
        
        distance_list = np.zeros(len(key_points))
        for i in range(len(key_points)):
            j = i + 1 if i + 1 < len(key_points) else 0
            tmp_vec = key_points[i].position - key_points[j].position
            distance_list[i] = np.linalg.norm(tmp_vec)
        distance_total = np.sum(distance_list)
        ratio_list = distance_list / distance_total
        
        if isinstance(num, Sequence):
            if len(num) == len(key_points):
                num_list = num
            else:
                raise ValueError("当前输入的 num 是一个数组, 需要保证 num 与 key_points 具有相同的长度.")
        else:  # 如果 num 不是一个数组, 则是总的采样点的数量
            # 计算每段路径的长度
            num_list = np.ceil(ratio_list * num).astype(int)
        distance_array = np.empty(0)
        for i in range(len(key_points)):
            distance_array = np.concatenate([distance_array,
                                             np.linspace(np.sum(distance_list[:i]),
                                                         np.sum(distance_list[:i + 1]),
                                                         num_list[i],
                                                         endpoint=False)])
        distance_array = np.append(distance_array, distance_total)
        
        # 初始化布洛赫波矢
        bloch_wave_vectors = np.zeros((0, 2))
        # 绕路径一周
        for i in range(len(key_points)):
            num = num_list[i]
            if i != len(key_points) - 1:
                new_vectors = np.linspace(key_points[i].position,
                                          key_points[i + 1].position,
                                          num, endpoint=False)
            else:
                new_vectors = np.linspace(key_points[-1].position,
                                          key_points[0].position,
                                          num, endpoint=False)
            bloch_wave_vectors = np.array([*bloch_wave_vectors,
                                           *new_vectors])
        # 为了首位相连, 将起点附加到最后一项
        bloch_wave_vectors = np.array([*bloch_wave_vectors,
                                       np.array(key_points[0].position)])
        
        bx = bloch_wave_vectors[:, 0]
        by = bloch_wave_vectors[:, 1]
        
        # 谐波轴
        p = np.arange(-np.floor(P / 2), np.floor(P / 2) + 1)
        q = np.arange(-np.floor(Q / 2), np.floor(Q / 2) + 1)
        
        # 卷积矩阵
        epsilon_r_conv = convmat(self.epsilon_r, P, Q)
        mu_r_conv = convmat(self.mu_r, P, Q)
        
        # 初始化标准化的频率数组
        
        omega: Sequence = np.zeros((spatial_harmonic_wave_num, np.shape(bloch_wave_vectors)[0]))
        
        for n, beta in enumerate(bloch_wave_vectors):
            Kx = bx[n] - 2 * np.pi * p / self.lattice_constant
            Ky = by[n] - 2 * np.pi * q / self.lattice_constant
            Kx, Ky = np.meshgrid(Kx, Ky)
            Kx, Ky = Kx.flatten(), Ky.flatten()
            Kx, Ky = np.diag(Kx), np.diag(Ky)
            
            if mode == "E":
                # 如果不是铁磁材料
                A = Kx ** 2 + Ky ** 2
                # 如果是铁磁材料
                # A  = Kx @ np.linalg.inv(URC) @ Kx + Ky @ np.linalg.inv(URC) @ Ky; # Operator for dielectric matrix
                
                k0_square = eigh(A, epsilon_r_conv, eigvals_only=True)
                k0_square = np.sort(k0_square)  # Sort eig vals (from lowest to highest)
                # norm = 2 * np.pi / self.Lx
                # k0 = np.real(np.sqrt(k0))   # Normalize eig-vals
                
                omega[:, n] = self.lattice_constant / (2 * np.pi) * np.real(np.sqrt(k0_square + 0j))
                # omega[:, n] = k0_square
            # TODO: H mode
            else:  # mode == "H"
                # A = Kx @ np.linalg.inv(ERC) @ Kx + Ky @ np.linalg.inv(ERC) @ Ky;
                #
                # if not params.is_magnetic:
                #     k0 = np.linalg.eigvals(A)
                # else:
                #     k0 = eigh(A, URC, eigvals_only=True);
                # k0 = np.sort(k0)
                # k0 = np.real(np.sqrt(k0)) / params.norm;
                # W[:, nbeta] = k0;
                pass
        if return_num_list:
            return distance_array, omega, num_list
        else:
            return distance_array, omega
    
    def solve_2D(self, P: int, Q: int, mode, bloch_wave_vectors: Sequence):
        # TODO: 画 3D 图时, bloch 波矢与结构有关, 应当自动生成
        
        # 总的空间谐波数
        spatial_harmonic_wave_num = P * Q
        
        # 布洛赫波矢
        bx = bloch_wave_vectors[0]
        by = bloch_wave_vectors[1]
        
        # 谐波轴
        p = np.arange(-np.floor(P / 2), np.floor(P / 2) + 1)
        q = np.arange(-np.floor(Q / 2), np.floor(Q / 2) + 1)
        
        # 卷积矩阵
        epsilon_r_conv = convmat(self.epsilon_r, P, Q)
        mu_r_conv = convmat(self.mu_r, P, Q)
        
        # 初始化标准化的频率数组
        
        omega: Sequence = np.zeros((spatial_harmonic_wave_num, *np.shape(bloch_wave_vectors)[1:]))
        for i in range(np.shape(bloch_wave_vectors)[1]):
            for j in range(np.shape(bloch_wave_vectors)[2]):
                # for n, beta in enumerate(bloch_wave_vectors):
                Kx = bx[i, j] - 2 * np.pi * p / self.lattice_constant
                Ky = by[i, j] - 2 * np.pi * q / self.lattice_constant
                Kx, Ky = np.meshgrid(Kx, Ky)
                Kx, Ky = Kx.flatten(), Ky.flatten()
                Kx, Ky = np.diag(Kx), np.diag(Ky)
                
                if mode == "E":
                    # 如果不是铁磁材料
                    A = Kx ** 2 + Ky ** 2
                    # 如果是铁磁材料
                    # A  = Kx @ np.linalg.inv(URC) @ Kx + Ky @ np.linalg.inv(URC) @ Ky; # Operator for dielectric matrix
                    
                    k0_square = eigh(A, epsilon_r_conv, eigvals_only=True)
                    k0_square = np.sort(k0_square)  # Sort eig vals (from lowest to highest)
                    # norm = 2 * np.pi / self.Lx
                    # k0 = np.real(np.sqrt(k0))   # Normalize eig-vals
                    
                    omega[:, i, j] = self.lattice_constant / (2 * np.pi) * np.real(np.sqrt(k0_square + 0j))
                    # omega[:, n] = k0_square
                # TODO: H mode
                else:  # mode == "H"
                    # A = Kx @ np.linalg.inv(ERC) @ Kx + Ky @ np.linalg.inv(ERC) @ Ky;
                    #
                    # if not params.is_magnetic:
                    #     k0 = np.linalg.eigvals(A)
                    # else:
                    #     k0 = eigh(A, URC, eigvals_only=True);
                    # k0 = np.sort(k0)
                    # k0 = np.real(np.sqrt(k0)) / params.norm;
                    # W[:, nbeta] = k0;
                    pass
        
        return omega
    
    def plot_path_band_diagram(self,
                               P: int,
                               Q: int,
                               mode,
                               key_points: Sequence[KeyPoint],
                               num: Union[int, Sequence] = 50,
                               show_bandgap=True):
        
        distance_array, omega, num_list = self.solve_path(P, Q, mode, key_points, num, return_num_list=True)
        
        tick_positions = np.zeros(len(key_points) + 1, dtype=int)
        tick_positions[0] = 0
        for i in range(1, len(num_list) + 1):
            tick_positions[i] = tick_positions[i - 1] + num_list[i - 1]
        
        tick_labels = []
        for key_point in key_points:
            tick_labels.append(key_point.name)
        tick_labels.append(key_points[0].name)
        
        # diagram along the path, 沿路径画图
        fig, ax = plt.subplots()
        
        for i in range(len(omega)):
            ax.plot(distance_array,
                    omega[i],
                    "k",
                    markerfacecolor="None",
                    # linewidth=1,
                    zorder=1)
        if show_bandgap:
            self.show_path_bandgap(ax, distance_array, omega)
        # plot settings
        ax.set_ylim(ymin=0)
        # 标注 key_point
        ax.vlines(distance_array[tick_positions[1:-1]], 0, ax.get_ylim()[1], "grey", "--", zorder=0)
        
        ax.set_xticks(distance_array[tick_positions], tick_labels)
        ax.set_xlim((0, distance_array[-1]))
        ax.set_ylabel("$k_0^2$")
        ax.set_xlabel("$\\vec \\beta$")
        ax.set_title("Path Band Diagram")
        return fig, ax
    
    def plot_2D_projection_band_diagram(self, P: int, Q: int, mode, bloch_wave_vectors: Sequence, level: int,
                                        cmap="rainbow"):
        omega = self.solve_2D(P, Q, mode, bloch_wave_vectors)
        fig, ax = plt.subplots()
        im = ax.pcolormesh(*bloch_wave_vectors, omega[level], linewidth=0, rasterized=True, cmap=cmap,
                           shading="gouraud")
        ax.set_box_aspect(1)
        cb = fig.colorbar(im, ax=ax)
        cb.ax.set_title("$\\omega$")
        self._set_ticks_range(ax, bloch_wave_vectors)
        ax.set_xlabel("$\\beta_x$")
        ax.set_ylabel("$\\beta_y$")
        ax.set_title("2D Projection Band Diagram")
        fig.savefig("./outputs/3D_projection_band_diagram.pdf")
    
    def plot_2D_band_diagram(self, P: int, Q: int, mode, bloch_wave_vectors: Sequence,
                             level: Union[int, Sequence[int]] = None):
        omega = self.solve_2D(P, Q, mode, bloch_wave_vectors)
        # TODO: x_array 和 y_array 与结构有关, 应当自动生成
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        
        if isinstance(level, int):  # TODO: 如果 level 是 int, 则画出投影等高线图
            ax.plot_surface(*bloch_wave_vectors, omega[level], alpha=.6, cmap="rainbow")
        elif isinstance(level, Iterable):  # FIXME: 没有判断可迭代对象的元素类型
            for level_i in level:
                ax.plot_surface(*bloch_wave_vectors, omega[level_i], alpha=.6)
        elif level is None:
            for i in range(len(omega)):
                ax.plot_surface(*bloch_wave_vectors, omega[i], alpha=.6)
        else:
            raise TypeError
        
        self._set_ticks_range(ax, bloch_wave_vectors)
        
        self._set_3d_ticks(fig, ax)
        ax.set_xlim(np.min(bloch_wave_vectors[0]),
                    np.max(bloch_wave_vectors[0]))
        ax.set_ylim(np.min(bloch_wave_vectors[1]),
                    np.max(bloch_wave_vectors[1]))
        ax.set_zlim(zmin=0)
        
        ax.set_xlabel("$\\beta_x$")
        ax.set_ylabel("$\\beta_y$")
        ax.set_zlabel("$\\omega$")
        ax.set_title("2D Band Diagram")
        fig.savefig("./outputs/3D_band_diagram.pdf")
    
    def _set_ticks_range(self, ax, bloch_wave_vectors):
        x_array, y_array = bloch_wave_vectors
        x_array = x_array[0]
        y_array = y_array[:, 0]
        
        x_extra_ticks = [np.min(x_array), np.max(x_array)]
        y_extra_ticks = [np.min(y_array), np.max(y_array)]
        
        ax.xaxis.set_tick_params(rotation=-90)
        
        ax.set_xticks(list(ax.get_xticks()) + x_extra_ticks)
        ax.set_yticks(list(ax.get_yticks()) + y_extra_ticks)
        
        ax.set_xlim((x_array[0], x_array[-1]))
        ax.set_ylim((y_array[0], y_array[-1]))
    
    def _set_3d_ticks(self, fig, ax):
        ax.xaxis.set_tick_params(rotation=45)
        ax.yaxis.set_tick_params(rotation=-15)
        ax.zaxis.set_tick_params(rotation=-15)
        ax.xaxis._axinfo["grid"]['linestyle'] = "--"
        ax.yaxis._axinfo["grid"]['linestyle'] = "--"
        ax.zaxis._axinfo["grid"]['linestyle'] = "--"
        ax.xaxis._axinfo["grid"]['linewidth'] = 0.5
        ax.yaxis._axinfo["grid"]['linewidth'] = 0.5
        ax.zaxis._axinfo["grid"]['linewidth'] = 0.5
        # print(dir(ax.xaxis))
        # dx = -1.
        # dy = 1.
        # offset = mpl.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        #
        # ax.xaxis.set_transform(ax.xaxis.get_transform() + offset)
        ax.tick_params(axis='both', which='major', labelsize=6)
        # ax.set
    
    def show_path_bandgap(self, ax, distance_array, omega, fineness=1e-3):
        min_array = np.min(omega, axis=1)
        max_array = np.max(omega, axis=1)
        recoder = []
        for i in range(np.shape(omega)[0] - 1):
            if min_array[i + 1] - max_array[i] > fineness:
                recoder.append([max_array[i], min_array[i + 1]])
        
        for item in recoder:
            rect = patches.Rectangle((0, item[0]),
                                     distance_array[-1],
                                     item[1] - item[0],
                                     facecolor="yellow",
                                     edgecolor="None",
                                     alpha=0.8,
                                     zorder=2)
            
            ax.add_patch(rect)
