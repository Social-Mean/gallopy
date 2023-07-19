import unittest

import numpy as np

from gallopy.mesh import MeshGenerator, plot_mesh


# from gallopy.fem import

class MyTestCase(unittest.TestCase):
    def test_regular_mesh(self):
        area_num = 200
        num = 10
        x = [0, 0, 1, 1]
        y = [0, 1, 0, 1]
        x1 = 0
        x2 = 1
        x1s = list(np.ones(num) * x1)[1:-1]
        x2s = list(np.ones(num) * x2)[1:-1]
        xs = list(np.linspace(0, 1, num))[1:-1]
        x_edge = x1s + x2s + xs + xs
        y_edge = xs + xs + x1s + x2s
        x += x_edge
        y += y_edge
        
        tri_mesh = MeshGenerator(x, y).regular_mesh()
        fig, ax = plot_mesh(tri_mesh)
        fig.savefig("./outputs/regular_mesh.pdf")
    
    def test_centroid_mesh(self):
        num_tri = 1000
        num = 10
        # num = int(np.floor(area_num**(1/2)))
        x = [0, 0, 1, 1]
        y = [0, 1, 0, 1]
        x1 = 0
        x2 = 1
        x1s = list(np.ones(num) * x1)[1:-1]
        x2s = list(np.ones(num) * x2)[1:-1]
        xs = list(np.linspace(0, 1, num))[1:-1]
        x_edge = x1s + x2s + xs + xs
        y_edge = xs + xs + x1s + x2s
        x += x_edge
        y += y_edge
        
        tri_mesh = MeshGenerator(x, y).centroid_mesh(num_tri)
        fig, ax = plot_mesh(tri_mesh)
        fig.savefig("./outputs/centroid_mesh.pdf")


if __name__ == '__main__':
    unittest.main()
