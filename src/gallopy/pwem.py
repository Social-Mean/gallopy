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


class KeyPoint(object):
    def __init__(self, key_point_name: str, key_point_position: ArrayLike):
        self.key_point_name = key_point_name
        self.key_point_position = key_point_position


class PWEMSolver(object):
    def __init__(self, epsilon_r, mu_r, lattice_constant):
        """
        
        :param epsilon_r:
        :param mu_r:
        :param bloch_wave_vectors:
        :param lattice_constant:
        """
        self.epsilon_r = epsilon_r
        self.mu_r = mu_r
        self.lattice_constant = lattice_constant
    
    # @property
    # def bloch_wave_vectors(self):
    #     return self._bloch_wave_vectors
    #
    # @bloch_wave_vectors.setter
    # def bloch_wave_vectors(self, ):
    
    def solve_path(self, P: int, Q: int, mode, key_points: Sequence[KeyPoint], num: int = 50):
        
        # 总的空间谐波数
        spatial_harmonic_wave_num = P * Q
        
        # 布洛赫波矢
        bloch_wave_vectors = np.zeros((0, 2))
        # 绕路径一周
        for i in range(len(key_points)):
            if i != len(key_points) - 1:
                new_vectors = np.linspace(key_points[i].key_point_position,
                                                       key_points[i + 1].key_point_position,
                                                       num)
            else:
                new_vectors = np.linspace(key_points[-1].key_point_position,
                                          key_points[0].key_point_position,
                                          num)
            bloch_wave_vectors = np.array([*bloch_wave_vectors,
                                           *new_vectors]
                                           )
            # bloch_wave_vectors.append(np.linspace(key_points[i].key_point_position,
            #                                             key_points[i + 1].key_point_position,
            #                                             num))
        
        # bloch_wave_vectors = np.append(bloch_wave_vectors, new_vectors
        #                                )
        # bloch_wave_vectors.append(np.linspace(key_points[-1].key_point_position,
        #                                            key_points[0].key_point_position,
        #                                            num))
        # 为了首位相连, 将起点附加到最后一项
        bloch_wave_vectors = np.array([*bloch_wave_vectors,
                                       np.array(key_points[0].key_point_position)])
        # bloch_wave_vectors.append(np.array([key_points[0].key_point_position]))
        # bloch_wave_vectors = np.array(bloch_wave_vectors)
        # bloch_wave_vector = np.concatenate([
        #     np.linspace(Gamma_point, X_point, Nn1),
        #     np.linspace(X_point, M_point, Nn2),
        #     np.linspace(M_point, Gamma_point, Nn3),
        #     [Gamma_point]])
        
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
        
        return omega
    
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
    
    def plot_path_band_diagram(self, P: int, Q: int, mode, key_points: Sequence[KeyPoint], num: int = 50):
        omega = self.solve_path(P, Q, mode, key_points, num)
        plt.figure()
        
        # 标注 key_point
        for i in range(len(key_points)):
            plt.vlines(num * i, 0, 0.65, "grey", "--")
        
        # diagram along a path, 沿路径画图
        for i in range(len(omega)):
            plt.plot(omega[i], "k")
        
        
        
        plt.ylim((0, 0.65))
        plt.xlim((0, np.shape(omega)[1] - 1))
        plt.ylabel("$k_0^2$")
        plt.xlabel("Array index of Bloch wave vector $\\vec \\beta$")
        plt.title("Path Band Diagram")
        plt.savefig("./outputs/2D_band_diagram.pdf")
    
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
        
        if isinstance(level, int):
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
