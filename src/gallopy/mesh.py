import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation, UniformTriRefiner


class MeshGenerator(object):
    def __init__(self, edge_x_arr, edge_y_arr):
        self.edge_x_arr = edge_x_arr
        self.edge_y_arr = edge_y_arr
    
    def regular_mesh(self):
        x, y = np.meshgrid(self.edge_x_arr, self.edge_y_arr)
        x = x.flatten()
        y = y.flatten()
        return Triangulation(x, y)
    
    def centroid_mesh(self, num_tri):
        triangulation = Triangulation(self.edge_x_arr, self.edge_y_arr)
        x = self.edge_x_arr.copy()
        y = self.edge_y_arr.copy()
        dif_min = 1 / num_tri  # / np.pi
        
        # while len(triangulation.triangles) < pt_num:
        # for _ in range(num_tri):
        while True:
            ns_mat = triangulation.triangles
            x_new = np.mean(triangulation.x[ns_mat], axis=1)
            y_new = np.mean(triangulation.y[ns_mat], axis=1)
            
            # TODO: 使用面积
            
            dif1 = (x_new - triangulation.x[ns_mat][:, 0]) ** 2 + (y_new - triangulation.y[ns_mat][:, 0]) ** 2
            dif2 = (x_new - triangulation.x[ns_mat][:, 1]) ** 2 + (y_new - triangulation.y[ns_mat][:, 1]) ** 2
            dif3 = (x_new - triangulation.x[ns_mat][:, 2]) ** 2 + (y_new - triangulation.y[ns_mat][:, 2]) ** 2
            #
            # dif1 = np.sqrt(dif1)
            # dif2 = np.sqrt(dif2)
            # dif3 = np.sqrt(dif3)
            
            dif = np.min(np.array([dif1, dif2, dif3]), axis=0)
            # dif = np.sum(np.array([dif1, dif2, dif3]), axis=0)
            # x_new = x_new[(dif1[dif1 > dif_min] + dif2[dif2 > dif_min] + dif3[dif3 > dif_min]) > 0]
            x_new = x_new[dif > dif_min]
            y_new = y_new[dif > dif_min]
            
            if len(x_new) == 0:
                break
            
            x += list(x_new)
            y += list(y_new)
            
            triangulation = Triangulation(x, y)
        return triangulation
    
    def uniform_mesh(self, triangulation: Triangulation = None):
        if triangulation is None:
            return UniformTriRefiner(Triangulation(self.edge_x_arr, self.edge_y_arr)).refine_triangulation()
        return UniformTriRefiner(triangulation).refine_triangulation()


def plot_mesh(triangulation: Triangulation, *, show_tag=False):
    fig, ax = plt.subplots()
    ax.triplot(triangulation, color="k", lw=.5)
    # plt.text(triangulation.x[triangulation.triangles[0]], triangulation.y[triangulation.triangles[0]], "a")
    if show_tag and np.shape(triangulation.triangles)[0] < 99:  # 如果网格过多, 也会强制不标tag
        for row_i, row in enumerate(triangulation.triangles):
            mid_x = np.mean(triangulation.x[row])
            mid_y = np.mean(triangulation.y[row])
            ax.text(mid_x, mid_y, row_i, color="r", ha="center", va="center")
            for col in row:
                ax.text(triangulation.x[col],
                        triangulation.y[col],
                        col,
                        ha="center",
                        va="center",
                        backgroundcolor="r",
                        color="w",
                        bbox=dict(boxstyle="circle"))
    
    ax.set_title("Triangular Mesh")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$", rotation=0)
    ax.set_xlim((triangulation.x.min(), triangulation.x.max()))
    ax.set_ylim((triangulation.y.min(), triangulation.y.max()))
    ax.set_box_aspect(1)
    return fig, ax
