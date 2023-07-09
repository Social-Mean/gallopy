import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["savefig.transparent"] = True
plt.rcParams["figure.dpi"] = 600
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["axes.prop_cycle"] = plt.cycler('color', ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"])
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
# 取消 3D 画图时坐标轴边界处的缓冲区
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new

