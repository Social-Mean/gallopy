# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:30:14 2023

@author: Social Mean 
@e-mail: 2052760@tongji.edu.cn
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks  # 寻峰函数

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["figure.autolayout"] = True
plt.rcParams["savefig.transparent"] = True
plt.rcParams["figure.dpi"] = 600
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1

c0 = 299_792_458


class TMMSolver(object):
    
    def __init__(self, omega, epsilon_list, mu_list, h_list, polarization):
        """
        sadsd
        """
        self.omega = omega
        self.epsilon_list = epsilon_list
        self.mu_list = mu_list
        self.h_list = h_list
        self.polarization = polarization
        self.N = len(epsilon_list)
        self.n_list = np.sqrt(self.epsilon_list * self.mu_list)
        self.k0 = self.omega / c0
    
    def get_M(self, kx):
        """
        asd
        :param kx:
        :return:
        """
        k = self.epsilon_list * self.mu_list * self.omega ** 2 / c0 ** 2
        kz = np.sqrt(
            self.epsilon_list * self.mu_list * self.omega ** 2 / c0 ** 2
            - kx ** 2
            + 0j
        )
        # 计算D矩阵
        D_list = []
        D_inv_list = []
        for i in range(self.N):
            if self.polarization == "TE":
                D_list.append(
                    np.array(
                        [
                            [1, 1],
                            [
                                kz[i] / (self.mu_list[i] * self.omega),
                                -kz[i] / (self.mu_list[i] * self.omega),
                            ],
                        ]
                    )
                )
                D_inv_list.append(
                    0.5
                    * np.array(
                        [
                            [1, (self.mu_list[i] * self.omega) / kz[i]],
                            [1, -(self.mu_list[i] * self.omega) / kz[i]],
                        ]
                    )
                )
            elif self.polarization == "TM":
                D_list.append(
                    np.array(
                        [
                            [
                                np.sqrt(
                                    self.epsilon_list[i] / self.mu_list[i]
                                ),
                                np.sqrt(
                                    self.epsilon_list[i] / self.mu_list[i]
                                ),
                            ],
                            [kz[i] / k[i], -kz[i] / k[i]],
                        ]
                    )
                )
                D_inv_list.append(
                    0.5
                    * np.array(
                        [
                            [
                                np.sqrt(
                                    self.mu_list[i] / self.epsilon_list[i]
                                ),
                                k[i] / kz[i],
                            ],
                            [
                                np.sqrt(
                                    self.mu_list[i] / self.epsilon_list[i]
                                ),
                                -k[i] / kz[i],
                            ],
                        ]
                    )
                )
        # 计算P矩阵
        P_list = []
        P_list.append(np.empty(2, dtype=complex))
        for i in range(1, self.N - 1):
            P_list.append(
                np.array(
                    [
                        [np.exp(1j * kz[i] * self.h_list[i]), 0],
                        [0, np.exp(-1j * kz[i] * self.h_list[i])],
                    ]
                )
            )
        P_list.append(np.empty(2, dtype=complex))
        # 计算 M 矩阵
        # M_list = [P_list[-2] @ np.linalg.inv(D_list[-2]) @ D_list[-1]]
        M_list = [P_list[-2] @ D_inv_list[-2] @ D_list[-1]]
        M = D_inv_list[0]
        for i in range(1, self.N - 1):
            M = M @ D_list[i] @ P_list[i] @ D_inv_list[i]
            # M_list.append(M @ D_list[i+1])
            # M_list.append(
            #     np.linalg.inv(D_list[i-1]) @ D_list[i] @ P_list[i] @ np.linalg.inv(D_list[i]) @ D_list[i+1]
            #     )
            # M_list.append(
            #     D_list[i] @ P_list[i] @ np.linalg.inv(D_list[i])
            #     )
            # M_list.append(M)
            # M_list.append(np.linalg.inv(D_list[i-1]) @ D_list[i])
            # M_list.append(np.linalg.inv(D_list[i]) @ D_list[i+1])
            # M_list.append(D_list[i] @ np.linalg.inv(D_list[i+1]))
            # M_list.append(D_list[i-1] @ np.linalg.inv(D_list[i]))
        for i in range(1, self.N - 2):
            M_list.append(
                P_list[self.N - 1 - i - 1]
                @ D_inv_list[self.N - 1 - i - 1]
                @ D_list[self.N - 1 - i]
                @ M_list[-1]
            )
        # M_list.append(
        #   inv(D_list[0]) @ D_list[1]
        #   )
        # M_list.append(M)
        M = M @ D_list[-1]
        # M_list.append(M)
        # M_list.append
        # print(len(M_list))
        # print("M_list\n", M_list)
        # print(len(M_list))
        M_list = np.array(M_list, dtype=object)
        # print(M_list)
        self.D_list = D_list
        self.P_list = P_list
        return M, M_list
    
    def get_neff(self, num=1000, plot=False, strictly=True):
        n_begin = max(self.n_list[0], self.n_list[-1])
        n_end = max(self.n_list)
        kx_tilde_list = np.linspace(
            n_begin * (1 + 1e-8), n_end * (1 - 1e-8), num
        )
        
        M11_abs_list = []
        M_all_list = []
        M_list_all_list = []
        for kx_tilde in kx_tilde_list:
            kx = kx_tilde * self.k0
            M, M_list = self.get_M(kx)
            M_all_list.append(M)
            M_list_all_list.append(M_list)
            M11_abs = abs(M[1, 1])
            M11_abs_list.append(M11_abs)
        M11_abs_list = np.array(M11_abs_list)
        peaks, _ = find_peaks(-M11_abs_list)
        # 删除 M11_abs 大于 1 时对应的有效折射率
        neff_list = []
        M_return_list = []
        M_list_return_list = []
        for i in range(len(peaks)):
            # 严格模式:
            if strictly:
                if M11_abs_list[peaks[i]] < 1:
                    neff_list.append(kx_tilde_list[peaks[i]])
                    M_return_list.append(M_all_list[peaks[i]])
                    M_list_return_list.append(M_list_all_list[peaks[i]])
            else:
                neff_list.append(kx_tilde_list[peaks[i]])
                M_return_list.append(M_all_list[peaks[i]])
                M_list_return_list.append(M_list_all_list[peaks[i]])
        
        # 画图
        if plot == True:
            plt.subplot().spines[["right", "top"]].set_visible(False)
            plt.plot(kx_tilde_list, M11_abs_list, "b")
            plt.scatter(kx_tilde_list[peaks], M11_abs_list[peaks], c="r")
            for neff in neff_list:
                plt.text(
                    neff,
                    0,
                    f"  $n_{{\mathrm{{eff}}}}={neff:.3f}$",
                    verticalalignment="top",
                    rotation=-70,
                    ha="left",
                    c="r",
                )
            
            # 画图的设置
            plt.ylim([0, 1])
            plt.xlim([n_begin, n_end])
            plt.xlabel("$k_x/k_0$")
            plt.ylabel("abs$(M_{11})$")
            plt.grid(True, linestyle="--")
            # plt.box(False)
            # plt.spines['top'].set_visible(False)
            # plt.spines['right'].set_visible(False)
            plt.title("Eigen Mode for " + self.polarization)
        
        # 将有效折射率从大到小排序
        # neff_list = np.flip(neff_list)
        # M_return_list = np.flip(M_return_list, axis=0)
        # M_list_return_list = np.flip(M_list_return_list, axis=0)
        neff_list = neff_list[::-1]
        M_return_list = M_return_list[::-1]
        M_list_return_list = M_list_return_list[::-1]
        # print("M_list_return_list", M_list_return_list)
        
        # print(M_return_list)
        
        # return neff_list, M11_abs_list, M_return_list, M_list_return_list[0,:]
        return neff_list, M11_abs_list, M_return_list, M_list_return_list
    
    # def get_field(self, neff, plot=False):
    def get_field(
            self,
            z_list,
            neff,
            M,
            eigen_M_list,
            plot=False,
            color="r",
            holdon=False,
            order=0,
    ):
        # print(M)
        kx_tilde = neff
        kx = kx_tilde * self.k0
        
        k_list = self.n_list * self.k0
        # # +0j 是为了给负数开方得到复数
        kz_list = np.sqrt(k_list ** 2 - kx ** 2 + 0j)
        # print("kz_list", kz_list)
        AB_list = [np.array([0, 1])]  # 是 [0, 1] 没错
        # AB_list = [np.array([1, -(kx**2 - self.k0**2*self.n_list[-1]**2)**(1/2)])]
        # for eigen_M in np.flip(eigen_M_list, axis=0):
        # print("eigen_M_list", eigen_M_list)
        for eigen_M in eigen_M_list:
            # for eigen_M in eigen_M_list:
            # print("size", (eigen_M))
            # print("AB_list[-1]", AB_list[-1])
            AB_list.append(eigen_M @ AB_list[0])
            # AB_list.append(eigen_M @ AB_list[0])
        # AB_list.append(M @ AB_list[0])
        if self.n_list[0] == self.n_list[-1]:
            AB_list.append(np.array([1, 0]))
        else:
            AB_list.append(M @ AB_list[0])
        # AB_list.append(inv(M) @ np.array([0, 1]))
        # AB_list.append(np.array([a, 1]))
        # print(AB_list)
        AB_list = np.array(AB_list)
        AB_list = AB_list[::-1]
        # print("AB_list\n", AB_list)
        # print("kz_list", kz_list)
        h_sum = sum(self.h_list[1:-1])
        E_list = []
        h_sum_list = [0]
        for i in range(1, self.N):
            h_sum_list.append(self.h_list[i] + h_sum_list[-1])
        for z in z_list:
            # 最左侧
            if z < 0:
                E_list.append(
                    # (AB_list[1][0] + AB_list[1][1]) * np.exp(-1j*kz_list[0]*z)
                    AB_list[0][0]
                    * np.exp(-1j * kz_list[0] * z)
                    # + AB_list[0][1] * np.exp(1j*kz_list[0]*z)
                )
            # 最右侧
            # elif z >= h_sum_list[-1]:
            #     z_relative = z - h_sum_list[-1]
            #     E_list.append(
            #       AB_list[-1][0] * np.exp(-1j*kz_list[-1]*z_relative) \
            #         + AB_list[-1][1] * np.exp(1j*kz_list[-1]*z_relative)
            #                 )
            # 中间
            else:
                for i in range(1, self.N):
                    if h_sum_list[i - 1] <= z < h_sum_list[i]:
                        z_relative = z - h_sum_list[i - 1]
                        E_list.append(
                            AB_list[i][0]
                            * np.exp(-1j * kz_list[i] * z_relative)
                            + AB_list[i][1]
                            * np.exp(1j * kz_list[i] * z_relative)
                        )
        
        E_list = np.array(E_list)
        # E_list = E_list / max(E_list)
        # E_list = np.abs(E_list)
        if plot == True:
            # shift = self.h_list[1] + 0.5 * self.h_list[2]
            shift = h_sum / 2
            # ylim = ax.get_ylim()
            if holdon:
                plt.plot(
                    z_list - shift,
                    np.abs(E_list) / np.max(np.abs(E_list)),
                    color + "-",
                    linewidth=2,
                    label="mode " + str(order),
                )
                plt.ylim((0, 1))
                # plt.xlim(((-2e-7, 2e-7)))
            else:
                fig, ax = plt.subplots()
                # ax = plt.gca()
                
                ax.plot(
                    z_list - shift,
                    np.abs(E_list) / np.max(np.abs(E_list)),
                    color + "-",
                )
                # ax.plot(z_list, np.abs(E_list) / np.max(np.abs(E_list)), "r-")
                
                # ax.set_ylim(ylim)
                ax.set_ylim((0, 1))
                # ax.set_xlim(z_list[0], z_list[-1])
                # ax.set_xlim(((-2e-7, 2e-7)))
            # HINT
            ### 测试
            # ax.set_ylim((0, 3))
            # ax.set_xlim(np.array([-.2, .2]) * 1e-5)
            # plt.plot(
            #   z_list,
            #   np.abs(AB_list[0][0] * np.exp(-1j*kz_list[0]*z_list) \
            #     + AB_list[0][1] * np.exp(-1j*kz_list[0]*z_list)),
            #   "g-"
            #   )
            for i in range(self.N - 1):
                # plt.vlines(h_sum_list[i], *ylim, linestyles="--", color="purple")
                # plt.vlines(h_sum_list[i]-shift, *ylim, linestyles="--", color="purple")
                # plt.vlines(h_sum_list[i]-shift, 0, 1, linestyles="--", color="violet", linewidth=1)
                plt.gca().add_patch(
                    mpl.patches.Rectangle(
                        (h_sum_list[i] - shift, 0),
                        h_sum_list[i + 1] - h_sum_list[i],
                        1,
                        color="antiquewhite",
                        alpha=(self.n_list[i + 1] - 1) / max(self.n_list),
                    )
                )
        return E_list


if __name__ == "__main__":
    
    def 王健平板波导():
        global lam0
        global n_list
        global h_list
        lam0 = 1.55e-6
        w = 5e-6
        n_list = np.array(
            [
                # 1.515, 2, 1
                # 1, 2, 1.515
                1.515,
                1.62,
                1
                # 3, 8, 1
            ]
        )
        h_list = np.array([np.inf, w, np.inf])
    
    
    def 五层槽波导vandervoid():
        global lam0
        global n_list
        global h_list
        
        lam0 = 1e-6
        # lam0 = 200
        w = 0.1 * lam0
        # d = .01 * lam0
        d = 0.025 * lam0
        n_clad = 8
        n_list = np.array([1, n_clad, 1, n_clad, 1])
        h_list = np.array([np.inf, w, d, w, np.inf])
    
    
    def 五层槽波导测试():
        global lam0
        global n_list
        global h_list
        
        lam0 = 589e-9
        
        n_list = np.array([1, 1.5, 1, 1.5, 1])
        
        h_list = np.array([np.inf, 180e-9, 50e-9, 180e-9, np.inf])
    
    
    n_list = 0
    lam0 = 0
    h_list = 0
    
    王健平板波导()
    # 五层槽波导vandervoid()
    # 五层槽波导测试()
    
    k0 = 2 * np.pi / lam0
    omega = 2 * np.pi * c0 / (1 * lam0)
    
    epsilon_list = n_list ** 2
    mu_list = np.ones(len(epsilon_list))
    polarization = "TM"
    h_sum = sum(h_list[1:-1])
    z_list = np.linspace(-h_sum, 2 * h_sum, 1000)
    neff, _, M, eigen_M_list = TMMSolver(
        omega, epsilon_list, mu_list, h_list, polarization
    ).get_neff(plot=True)
    # plt.savefig("neff.svg")
    print("neff", neff)
    order = 2
    # from 从lumerical导出的数据画图 import main
    
    plt.figure()
    # plt.plot(*main())
    for order, color in zip([0, 1], ["r", "b"]):
        # for order, color in zip([0,1,2,3], ["r", "g", "m", "y"]):
        TMMSolver(
            omega, epsilon_list, mu_list, h_list, polarization
        ).get_field(
            z_list,
            neff[order],
            M[order],
            eigen_M_list[order],
            plot=True,
            color=color,
            holdon=True,
            order=order,
        )
    # plt.xlim()
    plt.xlabel("$z$ / m")
    plt.ylabel("$\\frac{E}{\max(E)}$")
    plt.legend()
    print("M", M[order])
    plt.savefig("场分布三层.pdf")
    # plt.savefig("TMMmode-"+ polarization +".pdf")
    # plt.ylim((0, 0.3))
