import unittest
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../scr")
from gallopy.fem import FEMSolver1D, DirichletBoundaryCondition, FEMSolver2D, tripcolor, trisurface
from scipy.integrate import solve_bvp, odeint

from gallopy.mesh import MeshGenerator
from matplotlib.tri import UniformTriRefiner, Triangulation


def _triangulation():
    num_tri = 100
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
    mg = MeshGenerator(x, y)
    # triangulation = mg.regular_mesh()
    # triangulation = Triangulation(x, y)
    # 用重心法迭代创造网格
    triangulation = mg.centroid_mesh(num_tri)
    # 精细化网格
    triangulation = mg.uniform_mesh(triangulation)
    # 寻找当前节点下最优网格
    triangulation = Triangulation(triangulation.x, triangulation.y)
    return triangulation


class MyTestCase(unittest.TestCase):
    def test_fem1D(self):
        node_num = 50
        force_func = lambda x: x ** 3 + 1
        # force_func = 0
        alpha = -1
        # alpha = lambda x: -2*x+1
        beta = 1
        # beta = lambda x: 2 * x + 1
        
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
            dydt = [v, force_func(t) - u]
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
        plt.plot(analysis_result.x, analysis_result.y[0], label="SciPy.ode")
        # plt.plot(x_array, analysis_result, label="truth")
        plt.plot(x_array, result, "o", label="FEM", markerfacecolor="None", linewidth=0.1, markeredgecolor="k")
        plt.xlim((min(x_array), max(x_array)))
        # plt.ylim((min(result), max(result)))
        plt.legend()
        plt.title(r"$u_{xx}-u=x^3+1$")
        plt.savefig("./outputs/fem1D5.svg")
    
    def test_fem2D(self):
        section_num = 8
        node = np.zeros((3, 8))
    
    def test_mpl_tri(self):
        def create_random_mesh(pt_num):
            x = np.random.random(pt_num)
            y = np.random.random(pt_num)
            x = []
            y = []
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
            x1s = list(np.ones(num) * x1)[1:-1]
            x2s = list(np.ones(num) * x2)[1:-1]
            xs = list(np.linspace(0, 1, num))[1:-1]
            x_edge = x1s + x2s + xs + xs
            y_edge = xs + xs + x1s + x2s
            
            # x = 2*x - 1
            # y = 2*y - 1
            x += x_edge
            y += y_edge
            
            x = np.array(x)
            y = np.array(y)
            
            return mpl.tri.Triangulation(x, y)
        
        triangulation = _triangulation()
        # 圆比总和
        # print(TriAnalyzer(triangulation).circle_ratios().sum())
        
        # triangulation = create_regular_mesh(pt_num)
        
        # triangulation = mpl.tri.Triangulation(x, y)
        # print(triangulation.get_trifinder()(0, 0))
        # print(triangulation.neighbors)
        # print(triangulation.edges)
        # alpha_x = lambda x, y: 1
        f_func = lambda x, y: 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
        # f_func = lambda x, y: 1
        f_func = 0
        
        solver = FEMSolver2D(1, 1, 0, f_func, [])
        Phi = solver(triangulation)
        # solver.triangulation, _ = UniformTriRefiner(triangulation).refine_triangulation()
        
        fig, ax = solver.plot_mesh(show_tag=True)
        fig.savefig("./outputs/tri_mesh.pdf")
        fig.savefig("./outputs/tri_mesh.svg")
        
        fig, ax = solver.plot_K_mat()
        # fig.savefig("./outputs/K_mat.pdf")
        
        fig, ax = tripcolor(
            solver,
            show_mesh=False
        )
        # ax.axis(False)
        ax.set_title(r"$\nabla^2 𝛷 = 0$")
        # fig.savefig("./outputs/tripcolor.svg")
        # fig.savefig("./outputs/tripcolor.png")
        fig.savefig("./outputs/tripcolor.pdf")
        # fig.savefig("./outputs/tripcolor.jpg")
        
        fig, ax = trisurface(
            solver,
            # show_mesh=False
        )
        ax.set_title(r"$\nabla^2 𝛷 = 0$")
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
        
        fig3, _ = tripcolor(solver)
        fig3.savefig("./outputs/test_change_tri_fig3.pdf")
    
    def test_thermal_conduction_2D(self):
        filename = "thermal_conduction"
        triangulation = _triangulation()
        
        alpha_x = alpha_y = 1
        beta = 0
        f = 0
        fem = FEMSolver2D(alpha_x, alpha_y, beta, f, [])
        fem.triangulation = triangulation
        fig, ax = fem.plot_mesh()
        ax.set_title(r"2D Thermal Conduction $\nabla^2 𝛷 = 0$")
        fig.savefig(f"./outputs/{filename}_mesh.svg")
        
        fig, ax = tripcolor(fem, show_mesh=False)
        ax.set_title(r"2D Thermal Conduction $\nabla^2 𝛷 = 0$")
        fig.savefig(f"./outputs/{filename}_tripcolor.svg")
        
        fig, ax = tripcolor(fem)
        ax.set_title(r"2D Thermal Conduction $\nabla^2 𝛷 = 0$")
        fig.savefig(f"./outputs/{filename}_tripcolor_with_mesh.svg")
        
        fig, ax = trisurface(fem)
        ax.set_title(r"2D Thermal Conduction $\nabla^2 𝛷 = 0$")
        fig.savefig(f"./outputs/{filename}_trisurface.svg")
    
    def test_electric_field_2D(self):
        filename = "electric_field"
        title = r"2D Electric Field $\nabla^2 \phi = \rho/\varepsilon_0$"
        triangulation = _triangulation()
        
        alpha_x = alpha_y = 1
        beta = 0
        
        def f(x, y):
            pos_x = 0.5
            pos_y = 0.5
            radius = 0.1
            rho = 1
            if (x - pos_x) ** 2 + (y - pos_y) ** 2 < radius ** 2:
                return -rho
            return 0
        
        fem = FEMSolver2D(alpha_x, alpha_y, beta, f, [])
        fem.triangulation = triangulation
        fig, ax = fem.plot_mesh()
        ax.set_title(title)
        fig.savefig(f"./outputs/{filename}_mesh.svg")
        
        fig, ax = tripcolor(fem, show_mesh=False)
        ax.set_title(title)
        fig.savefig(f"./outputs/{filename}_tripcolor.svg")
        
        fig, ax = tripcolor(fem)
        ax.set_title(title)
        fig.savefig(f"./outputs/{filename}_tripcolor_with_mesh.svg")
        
        fig, ax = trisurface(fem)
        ax.set_title(title)
        fig.savefig(f"./outputs/{filename}_trisurface.svg")
    
    def test_wave_equation_with_time_1D(self):
        filename = "wave_equation"
        fmt = "pdf"
        title = r"Wave Equation $u_{tt}=v^2u_{xx}$"
        triangulation = _triangulation()
        v = 1
        alpha_x = v ** 2
        alpha_y = -1
        beta = 0
        
        f = 0
        
        fem = FEMSolver2D(alpha_x, alpha_y, beta, f, [])
        fem.triangulation = triangulation
        fig, ax = fem.plot_mesh()
        ax.set_ylabel("$t$")
        ax.set_title(title)
        fig.savefig(f"./outputs/{filename}_mesh.{fmt}")
        
        fig, ax = tripcolor(fem, show_mesh=False)
        ax.set_ylabel("$t$")
        ax.set_title(title)
        fig.savefig(f"./outputs/{filename}_tripcolor.{fmt}")
        
        fig, ax = tripcolor(fem)
        ax.set_ylabel("$t$")
        ax.set_title(title)
        fig.savefig(f"./outputs/{filename}_tripcolor_with_mesh.{fmt}")
        
        fig, ax = trisurface(fem)
        ax.set_ylabel("$t$")
        ax.set_title(title)
        fig.savefig(f"./outputs/{filename}_trisurface.{fmt}")


if __name__ == '__main__':
    unittest.main()
