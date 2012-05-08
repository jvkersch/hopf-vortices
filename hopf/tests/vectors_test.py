import sys
sys.path.append('..')

import unittest
import numpy as np
from vectors import is_float_equal, normalize


class testFloatEqual(unittest.TestCase):
    def setUp(self):
        "Set up random vector and small perturbations."

        N = 3

        from vectors import random_vector
        self.v = random_vector(N)
        self.w_real = random_vector(N)
        self.w_complex = random_vector(N, iscomplex=True)

    def testRealEquality(self):
        "Test floating-point equality of real vectors."

        v = self.v
        v_perturb = self.v + 1e-8*(self.w_real + 1)

        self.assertTrue(is_float_equal(v, v))
        self.assertTrue(is_float_equal(v, v_perturb)) 
        self.assertFalse(is_float_equal(v, v_perturb, tol=1e-10))
        self.assertFalse(is_float_equal(v, v + 1))

    def testComplexEquality(self):
        "Test floating-point equality of complex vectors."

        v = self.v
        v_perturb = self.v + 1e-8*self.w_complex

        self.assertTrue(is_float_equal(v, v_perturb, tol=1e-6))
        self.assertFalse(is_float_equal(v, v_perturb, tol=1e-9))        


class testNormalize(unittest.TestCase):

    def setUp(self):
        self.N = 17 # Number of rows to create -- esssentially arbitrary

        # Random complex matrix
        self.psi = np.random.rand(self.N, 2) + \
                1j*np.random.rand(self.N, 2)

        # Random real matrix of 6D vectors
        self.X = np.random.rand(self.N, 6)


    def testNormalizeComplex(self):
        psi_normalized = normalize(self.psi)
        
        for i in xrange(0, self.N):
            row = self.psi[i, :]
            row /= np.linalg.norm(row)
            self.assertTrue(is_float_equal(row, psi_normalized[i,:]))


    def testNormalizeReal(self):
        X_normalized = normalize(self.X)
        
        for i in xrange(0, self.N):
            row = self.X[i, :]
            row /= np.linalg.norm(row)
            self.assertTrue(is_float_equal(row, X_normalized[i,:]))
            

    def testNormalizeHardcoded(self):
        v1 = 1/2.**(1./2)*np.array([[1., 1., 0.]])
        v2 = normalize(v1)
        self.assertTrue(is_float_equal(v1[0, :], v2[0, :]))

        v3 = normalize(np.array([[5., 0., 0.]]))
        v4 = np.array([[1., 0., 0.]])
        self.assertTrue(is_float_equal(v3[0, :], v4[0, :]))


if __name__ == '__main__':
    unittest.main()
