import sys
sys.path.append('..')

import unittest
import numpy as np

from vectors import random_array                    
from su2_geometry import hatmap, is_SU
from lie_algebra import cayley

class testCayley(unittest.TestCase):

    def test_su_to_SU(self):

        N = 16
        a = random_array((N, 3))
        A = hatmap(a)
        U = cayley(A)

        self.assertTrue(is_SU(U))


if __name__ == '__main__':
    unittest.main()
