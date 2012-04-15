import sys
sys.path.append('..')

import unittest
import numpy as np
from vectors import (random_array, random_vector, hstack, 
                     is_float_equal, normalize)
from su2_geometry import hopf, inverse_hopf


class testHopf(unittest.TestCase):
    "Tests for Hopf fibration."

    def setUp(self):
        self.N = 17 # Number of rows to create -- esssentially arbitrary

        # Random complex matrix of unit-length vectors
        self.psi = random_array((self.N, 2), iscomplex=True)
        self.psi = normalize(self.psi)

        # Random real matrix of unit length vectors
        self.X = random_array((self.N, 3))
        self.X = normalize(self.X)

    def testPhase(self):
        """Check that the Hopf map is invariant under multiplication 
        by a random phase."""

        X = hopf(self.psi)

        theta = random_vector(self.N) # Column vector of phases
        phase = hstack(np.exp(1j*theta), 2)

        self.assertTrue(is_float_equal(X, hopf(phase*self.psi)))

    def testInverse(self):
        """Check that Hopf map composed with its inverse yield the identity."""

        # Hopf composed with the inverse should give the identity exactly
        X1 = hopf(inverse_hopf(self.X)) 
        self.assertTrue(is_float_equal(X1, self.X))

        # Inverse hopf composed with Hopf gives the identity up to phase
        psi1 = inverse_hopf(hopf(self.psi))
        phases = psi1/self.psi

        # Test whether columns of phase matrix are equal
        self.assertTrue(is_float_equal(phases[:, 0], phases[:, 1]))
        # Test whether column elements have unit norm
        p = phases[:, 0]
        self.assertTrue(is_float_equal(p*p.conjugate(), np.ones(self.N)))

if __name__ == '__main__':
    unittest.main()

