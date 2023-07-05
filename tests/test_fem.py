import unittest
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../scr")
from gallopy.fem import FEMSolver1D

class MyTestCase(unittest.TestCase):
    def test_fem1D(self):
        # M = 3
        force_func = lambda x: x + 1
        alpha = -1
        beta = 0
        solver = FEMSolver1D(alpha, beta, force_func)
        
        x_array = np.linspace(0, 1)
        result = solver.solve(x_array)
        analysis_result = x_array/3 + x_array**3/6 + x_array**2/2
        plt.plot(x_array, analysis_result, label="truth")
        plt.plot(x_array, result, "o", label="JLJ", markerfacecolor="None", linewidth=0.1, markeredgecolor="k")
        plt.show()
        
        


if __name__ == '__main__':
    unittest.main()
