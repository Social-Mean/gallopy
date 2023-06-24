import unittest
import sys

sys.path.append("../src")
from gallopy.tmm_mode import TMMSolver


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
