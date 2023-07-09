import unittest

import matplotlib.pyplot as plt

from gallopy.pwem import PWEMSolver, KeyPoint
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_path_band_diagram(self):
        # 圆柱形孔洞, a = 1, r = 0.35*a, epsilon_r = 9.0
        lattice_constant = 1
        radius = 0.35 * lattice_constant
        Nx = 500
        Ny = 500
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
        # 倒格矢
        T1 = np.array([2*np.pi / lattice_constant, 0])
        T2 = np.array([0, 2*np.pi / lattice_constant])

        # 重要的路径点
        Gamma_point = KeyPoint("$\\Gamma$", np.array([0, 0]))
        X_point = KeyPoint("$X$", T1 / 2)
        M_point = KeyPoint("$M$", T1 / 2 + T2 / 2)
        key_points = [Gamma_point, X_point, M_point]
        
 
        solver = PWEMSolver(epsilon_r, np.ones_like(epsilon_r), lattice_constant)
        P = 5
        Q = 5
        
        fig, ax = solver.plot_path_band_diagram(P, Q, "E", key_points, 1000)
        fig.savefig("./outputs/path_band_diagram.pdf")
    
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



        T1 = np.array([2 * np.pi / lattice_constant, 0])
        T2 = np.array([0, 2 * np.pi / lattice_constant])

        bloch_num = 50
        x_array = np.linspace(-T1[0]/2, T1[0]/2, bloch_num)
        y_array = np.linspace(-T2[1]/2, T2[1]/2, bloch_num)
        bloch_wave_vector = np.meshgrid(x_array, y_array)
        solver = PWEMSolver(epsilon_r, np.ones_like(epsilon_r), lattice_constant)
        P = 5
        Q = 5
        
        fig, ax = solver.plot_2D_band_diagram(P, Q, "E", bloch_wave_vector, [0, 1])
        fig.savefig("./outputs/2D_band_diagram.pdf")
        

 
    def test_2D_projection_band_diagram(self):
        # 圆柱形孔洞, a = 1, r = 0.35*a, epsilon_r = 9.0
        
        lattice_constant = 1
        radius = 0.35 * lattice_constant
        Nx = 500
        Ny = 500
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
        T1 = np.array([2 * np.pi / lattice_constant, 0])
        T2 = np.array([0, 2 * np.pi / lattice_constant])

        bloch_num = 50
        x_array = np.linspace(-T1[0]/2, T1[0]/2, bloch_num)
        y_array = np.linspace(-T2[1]/2, T2[1]/2, bloch_num)
        bloch_wave_vector = np.meshgrid(x_array, y_array)
        solver = PWEMSolver(epsilon_r, np.ones_like(epsilon_r), lattice_constant)
        P = 5
        Q = 5
        fig, ax = solver.plot_2D_projection_band_diagram(P, Q, "E", bloch_wave_vector, 0)
        fig.savefig("./outputs/2D_projection_band_diagram.pdf")
        
    def test_draw_structure(self):
        # 圆柱形孔洞, a = 1, r = 0.35*a, epsilon_r = 9.0
        lattice_constant = 1
        radius = 0.35 * lattice_constant
        Nx = 200
        Ny = 200
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
                    
        epsilon_r[10:20, 20:30] = -3
        
        epsilon_r[90:100, 20:80] = 15
        # 倒格矢
        T1 = np.array([2 * np.pi / lattice_constant, 0])
        T2 = np.array([0, 2 * np.pi / lattice_constant])
        
        # 重要的路径点
        Gamma_point = KeyPoint("$\\Gamma$", np.array([0, 0]))
        X_point = KeyPoint("$X$", T1 / 2)
        M_point = KeyPoint("$M$", T1 / 2 + T2 / 2)
        key_points = [Gamma_point, X_point, M_point]
        
        solver = PWEMSolver(epsilon_r, np.ones_like(epsilon_r), lattice_constant)
        P = 5
        Q = 5
        
        fig, ax = solver.draw_structure()
        fig.savefig("./outputs/structure.pdf")
        
        fig, ax = solver.draw_first_brillouin_zone(key_points)
        fig.savefig("./outputs/1BZ.pdf")
        
if __name__ == '__main__':
    unittest.main()
