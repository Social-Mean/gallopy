import unittest
import sys
import numpy as np
# sys.path.append("../src")
import matplotlib.pyplot as plt
from gallopy.tmm_mode import TMMSolver
c0 = 299_792_458

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here
        
        # def 王健平板波导():
        #     global lam0
        #     global n_list
        #     global h_list
        #     lam0 = 1.55e-6
        #     w = 5e-6
        #     n_list = np.array(
        #         [
        #             # 1.515, 2, 1
        #             # 1, 2, 1.515
        #             1.515,
        #             1.62,
        #             1
        #             # 3, 8, 1
        #         ]
        #     )
        #     h_list = np.array([np.inf, w, np.inf])
        #
        # def 五层槽波导vandervoid():
        #     global lam0
        #     global n_list
        #     global h_list
        #
        #     lam0 = 1e-6
        #     # lam0 = 200
        #     w = 0.1 * lam0
        #     # d = .01 * lam0
        #     d = 0.025 * lam0
        #     n_clad = 8
        #     n_list = np.array([1, n_clad, 1, n_clad, 1])
        #     h_list = np.array([np.inf, w, d, w, np.inf])
        #
        # def 五层槽波导测试():
        #     global lam0
        #     global n_list
        #     global h_list
        #
        #     lam0 = 589e-9
        #
        #     n_list = np.array([1, 1.5, 1, 1.5, 1])
        #
        #     h_list = np.array([np.inf, 180e-9, 50e-9, 180e-9, np.inf])
        #
        # n_list = 0
        # lam0 = 0
        # h_list = 0
        #
        # 王健平板波导()
        # # 五层槽波导vandervoid()
        # # 五层槽波导测试()
        #
        # k0 = 2 * np.pi / lam0
        # omega = 2 * np.pi * c0 / (1 * lam0)
        #
        # epsilon_list = n_list ** 2
        # mu_list = np.ones(len(epsilon_list))
        # polarization = "TM"
        # h_sum = sum(h_list[1:-1])
        # z_list = np.linspace(-h_sum, 2 * h_sum, 1000)
        # neff, _, M, eigen_M_list = TMMSolver(
        #     omega, epsilon_list, mu_list, h_list, polarization
        # ).get_neff(plot=True)
        # # plt.savefig("neff.svg")
        # print("neff", neff)
        # order = 2
        # # from 从lumerical导出的数据画图 import main
        #
        # plt.figure()
        # # plt.plot(*main())
        # for order, color in zip([0, 1], ["r", "b"]):
        #     # for order, color in zip([0,1,2,3], ["r", "g", "m", "y"]):
        #     TMMSolver(
        #         omega, epsilon_list, mu_list, h_list, polarization
        #     ).get_field(
        #         z_list,
        #         neff[order],
        #         M[order],
        #         eigen_M_list[order],
        #         plot=True,
        #         color=color,
        #         holdon=True,
        #         order=order,
        #     )
        # # plt.xlim()
        # plt.xlabel("$z$ / m")
        # plt.ylabel("$\\frac{E}{\max(E)}$")
        # plt.legend()
        # print("M", M[order])
        # plt.savefig("场分布三层.pdf")
        # # plt.savefig("TMMmode-"+ polarization +".pdf")
        # # plt.ylim((0, 0.3))


if __name__ == '__main__':
    unittest.main()
