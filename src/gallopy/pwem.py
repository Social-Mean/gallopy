from collections.abc import Iterable
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import ListedColormap
from numpy.typing import ArrayLike
from scipy.linalg import eigh

from . import physical_constant as const
from .matrix import convmat


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
                k0 = np.real(np.sqrt(k0_square + 0j))
                omega[:, n] = k0 * const.c0  # FIXME: 没有将 y_array 归一化到 [0, 1] 区间内
                # omega[:, n] = self.lattice_constant / (2 * np.pi) * np.real(np.sqrt(k0_square + 0j))
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
        
        omega: np.array = np.zeros((spatial_harmonic_wave_num, *np.shape(bloch_wave_vectors)[1:]))
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
                    k0 = np.real(np.sqrt(k0_square + 0j))
                    # omega[:, i, j] = self.lattice_constant / (2 * np.pi) * k0
                    omega[:, i, j] = k0 * const.c0
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
                               *,
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
        y_array = omega * self.lattice_constant / (2 * np.pi * const.c0)
        for i in range(len(y_array)):
            ax.plot(distance_array,
                    # omega[i] * self.lattice_constant / (2*np.pi * c0),
                    y_array[i],
                    "k",
                    markerfacecolor="None",
                    # linewidth=1,
                    zorder=1)
        if show_bandgap:
            self.show_path_bandgap(ax, distance_array, y_array)
        # plot settings
        ax.set_ylim(ymin=0)
        # 标注 key_point
        ax.vlines(distance_array[tick_positions[1:-1]], 0, ax.get_ylim()[1], "grey", "--", zorder=0)
        
        ax.set_xticks(distance_array[tick_positions], tick_labels)
        ax.set_xlim((0, distance_array[-1]))
        ax.set_xlabel("Bloch Wave Vector $\\vec \\beta$")
        ax.set_ylabel("Nomalized Frequency $\\frac{\\omega a}{2\\pi c_0}$")
        ax.set_title("Path Band Diagram")
        return fig, ax
    
    def plot_2D_projection_band_diagram(self,
                                        P: int,
                                        Q: int,
                                        mode,
                                        bloch_wave_vectors: Sequence,
                                        level: int,
                                        cmap="rainbow"):
        omega = self.solve_2D(P, Q, mode, bloch_wave_vectors)
        z_array = omega * self.lattice_constant / (2 * np.pi * const.c0)
        fig, ax = plt.subplots()
        im = ax.pcolormesh(*bloch_wave_vectors,
                           z_array[level],
                           linewidth=0,
                           rasterized=True,
                           cmap=cmap,
                           shading="gouraud",
                           vmin=0)
        ax.set_box_aspect(1)
        cb = fig.colorbar(im, ax=ax)
        cb.ax.set_title("$\\frac{\\omega a}{2\\pi c_0}$")
        # self._set_ticks_range(ax, bloch_wave_vectors)
        tmp_len = np.pi / self.lattice_constant
        ax.set_xticks([-tmp_len, 0, tmp_len], ["$-\\dfrac{\\pi}{a}$", "$0$", "$\\dfrac{\\pi}{a}$"])
        ax.set_yticks([-tmp_len, 0, tmp_len], ["$-\\dfrac{\\pi}{a}$", "$0$", "$\\dfrac{\\pi}{a}$"])
        
        ax.set_xlabel("$\\beta_x$")
        ax.set_ylabel("$\\beta_y$", rotation=0)
        ax.set_title(f"2D Projection Band Diagram of Level {level}")
        # fig.savefig("./outputs/3D_projection_band_diagram.pdf")
        return fig, ax
    
    def plot_2D_band_diagram(self, P: int, Q: int, mode, bloch_wave_vectors: Sequence,
                             level: Union[int, Sequence[int]] = None):
        omega = self.solve_2D(P, Q, mode, bloch_wave_vectors)
        z_array = omega * self.lattice_constant / (2 * np.pi * const.c0)
        # TODO: x_array 和 y_array 与结构有关, 应当自动生成
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set_proj_type("ortho")
        
        if isinstance(level, int):  # TODO: 如果 level 是 int, 则画出投影等高线图
            ax.plot_surface(*bloch_wave_vectors, z_array[level], alpha=.6, cmap="rainbow")
        elif isinstance(level, Iterable):  # FIXME: 没有判断可迭代对象的元素类型
            for level_i in level:
                ax.plot_surface(*bloch_wave_vectors, z_array[level_i], alpha=.6)
        elif level is None:
            for i in range(len(z_array)):
                ax.plot_surface(*bloch_wave_vectors, z_array[i], alpha=.6)
        else:
            raise TypeError
        
        # self._set_ticks_range(ax, bloch_wave_vectors)
        tmp_len = np.pi / self.lattice_constant
        ax.set_xticks([-tmp_len, 0, tmp_len], ["$-\\dfrac{\\pi}{a}$", "$0$", "$\\dfrac{\\pi}{a}$"])
        ax.set_yticks([-tmp_len, 0, tmp_len], ["$-\\dfrac{\\pi}{a}$", "$0$", "$\\dfrac{\\pi}{a}$"])
        
        ax.set_xlim3d((-tmp_len, tmp_len))
        ax.set_ylim3d((-tmp_len, tmp_len))
        
        ax.set_zlim3d(zmin=0)
        ax.azim = 225
        
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            # backgrounds
            axis.pane.fill = False
            axis.set_pane_color((1.0, 1.0, 1.0, 0))
            # grid settings
            axis._axinfo["grid"]['linestyle'] = "--"
            axis._axinfo["grid"]['linewidth'] = 0.5
            axis.set_rotate_label(False)
        
        # labels
        ax.set_xlabel("$\\beta_x$", rotation=0)
        ax.set_ylabel("$\\beta_y$", rotation=0)
        ax.set_zlabel("$\\frac{\\omega a}{2\\pi c_0}$", rotation=0)
        
        ax.set_title(f"2D Band Diagram of Level {level}")
        # fig.savefig("./outputs/3D_band_diagram.pdf")
        return fig, ax
    
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
    
    def show_path_bandgap(self, ax, distance_array, y_array, fineness=1e-3):
        min_array = np.min(y_array, axis=1)
        max_array = np.max(y_array, axis=1)
        recoder = []
        for i in range(np.shape(y_array)[0] - 1):
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
    
    def draw_structure(self, cmap=ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"])):
        Nx, Ny = np.shape(self.epsilon_r)
        half_lattice_constant = self.lattice_constant / 2
        x_array = np.linspace(-half_lattice_constant, half_lattice_constant, Nx)
        y_array = np.linspace(-half_lattice_constant, half_lattice_constant, Ny)
        
        fig, ax = plt.subplots()
        
        # im = ax.pcolormesh(x_array, y_array, self.epsilon_r, cmap="Greys")
        im = ax.pcolormesh(x_array, y_array, self.epsilon_r.transpose(),
                           cmap=cmap,
                           linewidth=0,
                           rasterized=True)
        ax.set_box_aspect(1)
        ax.set_xticks([x_array[0], 0, x_array[-1]], [r"$-a/2$", "0", r"$a/2$"])
        ax.set_yticks([y_array[0], 0, y_array[-1]], [r"$-a/2$", "0", r"$a/2$"])
        ax.set_xlim((x_array[0], x_array[-1]))
        ax.set_ylim((y_array[0], y_array[-1]))
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$", rotation=0)
        ax.set_title("Diagram of the Lattice Structure")
        
        cb = fig.colorbar(im, ax=ax)
        cb.ax.set_title("$\\epsilon_r$")
        
        white = np.array([1, 1, 1])
        black = np.array([0, 0, 0])
        # 添加最大和最小值的刻度
        # cb.set_ticks(list(cb.get_ticks()) + [np.min(self.epsilon_r), np.max(self.epsilon_r)])
        
        ticks = np.sort(np.array(cb.get_ticks()))
        color_num = len(ticks) - 1
        # ticks_normalized = ticks - min(ticks)
        # ticks_normalized = ticks_normalized / max(ticks_normalized)
        # color_arr = []
        # for tick in ticks_normalized:
        #     color_arr.append([1-tick, 1-tick, 1-tick, 1])
        # color_arr = color_arr[1:]
        
        color_arr = np.linspace(white, black, color_num)
        cmap = ListedColormap(color_arr)
        im.set_cmap(cmap)
        cb.set_ticks(ticks)
        im.set_clim((min(ticks), max(ticks)))
        
        return fig, ax
    
    def draw_first_brillouin_zone(self, key_points):
        half_len = np.pi / self.lattice_constant
        rect_1BZ = patches.Rectangle((-half_len, -half_len),
                                     2 * half_len,
                                     2 * half_len,
                                     color="None")
        
        fig, ax = plt.subplots()
        ax.set_box_aspect(1)
        ax.add_patch(rect_1BZ)
        
        pos_list = [_.position for _ in key_points]
        name_list = [_.name for _ in key_points]
        
        # polygon
        xy_arr = np.zeros((len(key_points), 2))
        for i in range(len(key_points)):
            xy_arr[i] = key_points[i].position
        polygon = patches.Polygon(xy_arr, color="#8eb4e3", alpha=.5, clip_on=False)
        ax.add_patch(polygon)
        
        # points and texts
        for point in key_points:
            circ = patches.Circle(point.position, 0.1, clip_on=False)
            ax.add_patch(circ)
            ax.text(*point.position, point.name, ha="right", va="bottom")
        
        # arrow
        for i in range(len(key_points)):
            x, y = key_points[i].position
            j = i + 1 if i + 1 < len(key_points) else 0
            dx, dy = key_points[j].position - key_points[i].position
            arrow = patches.Arrow(x, y, dx, dy, 0.1)
            # ax.add_patch(arrow)
            ax.arrow(x, y, dx, dy, width=.03, length_includes_head=True, facecolor="k", clip_on=False)
        
        ax.set_xticks([-half_len, 0, half_len], [r"$-\dfrac{\pi}{a}$", "0", r"$\dfrac{\pi}{a}$"])
        ax.set_yticks([-half_len, 0, half_len], [r"$-\dfrac{\pi}{a}$", "0", r"$\dfrac{\pi}{a}$"])
        
        ax.set_xlim((-half_len, half_len))
        ax.set_ylim((-half_len, half_len))
        
        ax.set_title("Diagram of the First Brillouin Zone")
        ax.set_xlabel(r"$\beta_x$")
        ax.set_ylabel(r"$\beta_y$", rotation=0)
        
        return fig, ax
