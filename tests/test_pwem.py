import unittest

import matplotlib.pyplot as plt

from gallopy.pwem import PWEMSolver
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_2D_band_diagram(self):
        # 圆柱形孔洞, a = 1, r = 0.35*a, epsilon_r = 9.0
        
        lattice_constant = 1
        radius = 0.35 * lattice_constant
        Nx = 101
        Ny = 101
        epsilon_2 = 9.0
        
        x_step = lattice_constant / Nx
        y_step = lattice_constant / Ny
        
        center_pos = np.array([lattice_constant / 2, lattice_constant / 2])
        
        # 构造 epsilon_r
        epsilon_r = np.ones((Nx, Ny))
        for i in range(Nx):
            for j in range(Ny):
                this_pos = np.array([i * x_step, j * y_step])
                distance_from_center = np.linalg.norm(this_pos - center_pos)
                if distance_from_center > radius:
                    epsilon_r[i, j] = epsilon_2

        
        # 构造布洛赫波矢量
        # 布洛赫矢量的总路程
        path_distance = (1 + np.sqrt(2) / 2) * lattice_constant
        # 布洛赫矢量取点的数量
        # TODO 按照路径的相对长短自动分配取点的数量
        Nn_each = 100
        Nn1 = Nn_each
        Nn2 = Nn_each
        Nn3 = int(np.floor(1.5*Nn_each))
        # 布洛赫矢量的步长
        # bloch_step = path_distance / Nn
        # 计算倒格矢
        # t1 = np.array([lattice_constant, 0])
        # t2 = np.array([0, lattice_constant])
        T1 = np.array([2*np.pi / lattice_constant, 0])
        T2 = np.array([0, 2*np.pi / lattice_constant])
        # 重要的路径点
        Gamma_point = np.array([0, 0])
        X_point = T1 / 2
        M_point = T1 / 2 + T2 / 2
        
        bloch_wave_vector = np.concatenate([
            np.linspace(Gamma_point, X_point, Nn1),
            np.linspace(X_point, M_point, Nn2),
            np.linspace(M_point, Gamma_point, Nn3),
            [Gamma_point]])
        # print(bloch_wave_vector)
        a = PWEMSolver(epsilon_r, np.ones_like(epsilon_r), lattice_constant)
        P = 5
        Q = 5
        a.solve_2D(P, Q, "E", bloch_wave_vector)
        a.plot_2D_band_diagram()
    
    def test_3D_band_diagram(self):
        # 圆柱形孔洞, a = 1, r = 0.35*a, epsilon_r = 9.0
        
        lattice_constant = 1
        radius = 0.35 * lattice_constant
        Nx = 101
        Ny = 101
        epsilon_2 = 9.0
        
        x_step = lattice_constant / Nx
        y_step = lattice_constant / Ny
        
        center_pos = np.array([lattice_constant / 2, lattice_constant / 2])
        
        # 构造 epsilon_r
        epsilon_r = np.ones((Nx, Ny))
        for i in range(Nx):
            for j in range(Ny):
                this_pos = np.array([i * x_step, j * y_step])
                distance_from_center = np.linalg.norm(this_pos - center_pos)
                if distance_from_center > radius:
                    epsilon_r[i, j] = epsilon_2
        
        # 构造布洛赫波矢量
        # 布洛赫矢量的总路程
        path_distance = (1 + np.sqrt(2) / 2) * lattice_constant
        # 布洛赫矢量取点的数量
        # TODO 按照路径的相对长短自动分配取点的数量
        Nn_each = 100
        Nn1 = Nn_each
        Nn2 = Nn_each
        Nn3 = int(np.floor(1.5 * Nn_each))
        # 布洛赫矢量的步长
        # bloch_step = path_distance / Nn
        # 计算倒格矢
        # t1 = np.array([lattice_constant, 0])
        # t2 = np.array([0, lattice_constant])
        T1 = np.array([2 * np.pi / lattice_constant, 0])
        T2 = np.array([0, 2 * np.pi / lattice_constant])
        # 重要的路径点
        Gamma_point = np.array([0, 0])
        X_point = T1 / 2
        M_point = T1 / 2 + T2 / 2
        
        # bloch_wave_vector = np.concatenate([
        #     np.linspace(Gamma_point, X_point, Nn1),
        #     np.linspace(X_point, M_point, Nn2),
        #     np.linspace(M_point, Gamma_point, Nn3),
        #     [Gamma_point]])
        bloch_num = 100
        x_array = np.linspace(-T1[0], T1[0], bloch_num)
        y_array = np.linspace(-T2[1], T2[1], bloch_num)
        bloch_wave_vector = np.meshgrid(x_array, y_array)
        a = PWEMSolver(epsilon_r, np.ones_like(epsilon_r), lattice_constant)
        P = 5
        Q = 5
        a.solve_3D(P, Q, "E", bloch_wave_vector)
        a.plot_3D_band_diagram(x_array, y_array, 1)
        # a.plot_3D_projection_band_diagram(0)
        # a.plot_3D_projection_band_diagram(1)
        # a.plot_3D_projection_band_diagram(2)
        # a.plot_3D_band_diagram(x_array, y_array, "asd")
        a.plot_3D_band_diagram(x_array, y_array, np.array([1, 4]))
        # self.assertRaises(TypeError, plot_3D_band_diagram(x_array, y_array, "sad"))


if __name__ == '__main__':
    unittest.main()
