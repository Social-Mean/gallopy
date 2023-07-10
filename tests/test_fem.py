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
        force_func = lambda x: x**2+1
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
            
            return [u0, u1-0]
        
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
        # x = np.random.random(20)
        # y = np.random.random(20)
        # x = np.array([0, 0, 2, 2])
        # y = np.array([0, 1, 0, 1])
        x = np.linspace(0, 10, 3)
        y = np.linspace(0, 10, 3)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        # x[3] = 7
        # y[3] = 9
        
        triangulation = mpl.tri.Triangulation(x, y)
        print(triangulation.triangles)
        print(triangulation.x, triangulation.y)
        plt.triplot(triangulation)
        plt.savefig("./outputs/tri_mesh.pdf")
        
        solver = FEMSolver2D(1, 1, 0, 0, [])
        solver.solve(triangulation)


if __name__ == '__main__':
    unittest.main()
