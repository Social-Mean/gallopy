import unittest

import numpy as np

from gallopy.matrix import kronecker_delta


class MyTestCase(unittest.TestCase):
    def test_kronecker_delta(self):
        # 数字与数字的判断
        self.assertEqual(kronecker_delta(1, 1), 1)
        self.assertEqual(kronecker_delta(1, 0), 0)
        
        # 矩阵与矩阵的判断
        a_mat = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])
        b_mat = a_mat.transpose()
        bool_mat = (kronecker_delta(a_mat, b_mat) - np.eye(3)).all()
        self.assertFalse(bool_mat)
        
        # 矩阵与数的判断
        bool_mat_2 = (kronecker_delta(a_mat, 1) - np.array([[1, 0, 0],
                                                            [0, 0, 0],
                                                            [0, 0, 0]])).all()
        self.assertFalse(bool_mat_2)
        # TODO: 写明预期结果, 提高易读性


if __name__ == '__main__':
    unittest.main()
