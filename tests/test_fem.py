import random
import unittest
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append("../scr")
from gallopy.fem import FEMSolver1D, DirichletBoundaryCondition, FEMSolver2D
from gallopy import rcParams
from scipy.integrate import solve_bvp, odeint


class MyTestCase(unittest.TestCase):
    def test_fem1D(self):
        node_num = 50
        force_func = lambda x: x ** 2 + 1
        alpha = -1
        # alpha = lambda x: -2*x+1
        beta = 0
        # beta = lambda x: -2 * x - 1
        
        x_array = np.linspace(-3, 3, node_num)
        # x_array = np.concatenate([np.linspace(0, 0.5, 10, endpoint=False),
        #                           np.linspace(0.5, 1, node_num)])
        
        condition = [
            DirichletBoundaryCondition(x_array[0], 0),
            DirichletBoundaryCondition(x_array[-1], 0),
            # DirichletBoundaryCondition(0.3, 0.5),
        ]
        # condition.append(DirichletBoundaryCondition(x_array[20], .6))
        
        solver = FEMSolver1D(alpha, beta, force_func, condition)
        
        result = solver.solve(x_array)
        
        ########## 解析解
        def func(t, y):
            u, v = y
            dydt = [v, force_func(t)]
            return dydt
        
        def bc(y0, y1):
            u0, v0 = y0
            u1, v1 = y1
            
            return [u0, u1 - 0]
        
        t = x_array
        
        ystart = odeint(func, [0, 1], t, tfirst=True)
        analysis_result = solve_bvp(func, bc, t, ystart.T)
        
        # analysis_result = x_array / 3 + x_array ** 3 / 6 + x_array ** 2 / 2
        
        ########## 解析解
        
        plt.subplots()
        plt.plot(analysis_result.x, analysis_result.y[0], label="truth")
        # plt.plot(x_array, analysis_result, label="truth")
        plt.plot(x_array, result, "o", label="JLJ", markerfacecolor="None", linewidth=0.1, markeredgecolor="k")
        plt.xlim((min(x_array), max(x_array)))
        # plt.ylim((min(result), max(result)))
        plt.savefig("./outputs/test_fem1D.pdf")
    
    def test_fem2D(self):
        section_num = 8
        node = np.zeros((3, 8))
    
    def test_mpl_tri(self):
        def create_random_mesh(pt_num):
            x = np.random.random(pt_num)
            y = np.random.random(pt_num)
    
            x = list(x)
            y = list(y)
    
            x.append(0)
            y.append(0)
    
            x.append(0)
            y.append(1)
    
            x.append(1)
            y.append(0)
    
            x.append(1)
            y.append(1)
            

            
            x1 = 0
            x2 = 1
            num = int(np.floor(np.sqrt(pt_num)))
            x1s = list(np.ones(num)*x1)[1:-1]
            x2s = list(np.ones(num)*x2)[1:-1]
            xs = list(np.linspace(0, 1, num))[1:-1]
            x_edge = x1s + x2s + xs + xs
            y_edge = xs + xs + x1s + x2s
            
            # x = 2*x - 1
            # y = 2*y - 1
            x += x_edge
            y += y_edge
            
            x = np.array(x)
            y = np.array(y)
            return x, y
        
        def create_regular_mesh(pt_num):
            num = int(np.floor(np.sqrt(pt_num)))
            x = np.linspace(0, 1, num)
            y = np.linspace(0, 1, num)
            x, y = np.meshgrid(x, y)
            x = x.flatten()
            y = y.flatten()
            return x, y
        
        pt_num = 10000
        
        x, y = create_random_mesh(pt_num)
        
        # x, y = create_regular_mesh(pt_num)
        
        triangulation = mpl.tri.Triangulation(x, y)
        # print(triangulation.get_trifinder()(0, 0))
        # print(triangulation.neighbors)
        # print(triangulation.edges)
        # alpha_x = lambda x, y: 1
        f_func = lambda x, y: 2*np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)
        # f_func = lambda x, y: 1
        f_func = 0

        
        solver = FEMSolver2D(1, 1, 0, f_func, [])
        Phi = solver(triangulation)
        solver.triangulation = triangulation
        
        fig, ax = solver.plot_mesh(show_tag=True)
        fig.savefig("./outputs/tri_mesh.pdf")
        
        fig, ax = solver.plot_K_mat()
        fig.savefig("./outputs/K_mat.pdf")
        
        fig, ax = solver.tripcolor(
            show_mesh=False
        )
        fig.savefig("./outputs/Phi.pdf")
        
        fig, ax = solver.trisurface(
            # show_mesh=False
        )
        fig.savefig("./outputs/trisurface.pdf")
    
    def test_change_tri(self):
        pt_num = 20
        x = np.random.random(pt_num)
        y = np.random.random(pt_num)
        
        x = list(x)
        y = list(y)
        
        x.append(0)
        y.append(0)
        
        x.append(0)
        y.append(1)
        
        x.append(1)
        y.append(0)
        
        x.append(1)
        y.append(1)
        
        tri1 = mpl.tri.Triangulation(x, y)
        solver = FEMSolver2D(1, 1, 0, 1, [])
        solver.triangulation = tri1
        fig1, _ = solver.plot_mesh()
        
        x = np.random.random(pt_num)
        y = np.random.random(pt_num)
        
        x = list(x)
        y = list(y)
        
        x.append(0)
        y.append(0)
        
        x.append(0)
        y.append(1)
        
        x.append(1)
        y.append(0)
        
        x.append(1)
        y.append(1)
        
        tri2 = mpl.tri.Triangulation(x, y)
        solver.triangulation = tri2
        fig2, _ = solver.plot_mesh()
        
        fig1.savefig("./outputs/tri1.pdf")
        fig2.savefig("./outputs/tri2.pdf")
        
        # solver.solve()
        
        fig3, _ = solver.tripcolor()
        fig3.savefig("./outputs/test_change_tri_fig3.pdf")


if __name__ == '__main__':
    unittest.main()
