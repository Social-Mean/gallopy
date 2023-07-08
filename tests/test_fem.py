import unittest
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../scr")
from gallopy.fem import FEMSolver1D, DirichletBoundaryCondition
from gallopy import rcParams


class MyTestCase(unittest.TestCase):
    def test_fem1D(self):
        node_num = 50
        force_func = lambda x: x + 1
        alpha = -1
        beta = 0
        
        x_array = np.linspace(0, 1, node_num)
        # x_array = np.concatenate([np.linspace(0, 0.5, 10, endpoint=False),
        #                           np.linspace(0.5, 1, node_num)])
        
        condition = [
            DirichletBoundaryCondition(0, 0),
            DirichletBoundaryCondition(1, 1),
            # DirichletBoundaryCondition(0.3, 0.5),
        ]
        # condition.append(DirichletBoundaryCondition(x_array[20], .6))
        
        solver = FEMSolver1D(alpha, beta, force_func, condition)
        
        result = solver.solve(x_array)
        analysis_result = x_array / 3 + x_array ** 3 / 6 + x_array ** 2 / 2
        
        plt.subplots()
        plt.plot(x_array, analysis_result, label="truth")
        plt.plot(x_array, result, "o", label="JLJ", markerfacecolor="None", linewidth=0.1, markeredgecolor="k")
        plt.xlim((min(x_array), max(x_array)))
        plt.ylim((min(analysis_result), max(analysis_result)))
        plt.savefig("./outputs/test_fem1D.pdf")
    
    def test_fem2D(self):
        section_num = 8
        node = np.zeros((3, 8))


if __name__ == '__main__':
    unittest.main()
