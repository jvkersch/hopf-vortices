import sys
sys.path.append('..')

import unittest
import numpy as np

from vectors import random_array, is_float_equal                  
from su2_geometry import hatmap, is_SU, cayley_klein
from lie_algebra import cayley


class testCayley(unittest.TestCase):

    def test_su_to_SU(self):
        """Test that the image of generic Cayley map ends up in SU(2)."""

        N = 16
        a = random_array((N, 3))
        A = hatmap(a)
        U = cayley(A)

        self.assertTrue(is_SU(U).all())


    def testCayleyKlein(self):
        """Test whether generic Cayley and Cayley-Klein agree on su(2)."""

        N = 16
        a = random_array((N, 3))
        A = hatmap(a)
     
        U1 = cayley(A)
        U2 = cayley_klein(a)

        self.assertTrue(is_float_equal(U1, U2))


if __name__ == '__main__':
    unittest.main()
