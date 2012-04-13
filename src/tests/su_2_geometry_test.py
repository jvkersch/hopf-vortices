import sys
sys.path.append('..')

import unittest
import su2_geometry
import numpy as np

def is_float_vector_equal(v1, v2):
    tol = 1e-8
    return all(np.abs(v1 - v2) < tol)


class testNormalize(unittest.TestCase):

    def setUp(self):
        self.N = 17 # Number of rows to create -- esssentially arbitrary

        # Random complex matrix
        self.psi = np.random.rand(self.N, 2) + \
                1j*np.random.rand(self.N, 2)

        # Random real matrix of 6D vectors
        self.X = np.random.rand(self.N, 6)


    def testNormalizeComplex(self):
        psi_normalized = su2_geometry.normalize(self.psi)
        
        for i in xrange(0, self.N):
            row = self.psi[i, :]
            row /= np.linalg.norm(row)
            self.assertTrue(is_float_vector_equal(row, psi_normalized[i,:]))


    def testNormalizeReal(self):
        X_normalized = su2_geometry.normalize(self.X)
        
        for i in xrange(0, self.N):
            row = self.X[i, :]
            row /= np.linalg.norm(row)
            self.assertTrue(is_float_vector_equal(row, X_normalized[i,:]))
            

    def testNormalizeHardcoded(self):
        v1 = 1/2.**(1./2)*np.array([[1., 1., 0.]])
        v2 = su2_geometry.normalize(v1)
        self.assertTrue(is_float_vector_equal(v1[0, :], v2[0, :]))

        v3 = su2_geometry.normalize(np.array([[5., 0., 0.]]))
        v4 = np.array([[1., 0., 0.]])
        self.assertTrue(is_float_vector_equal(v3[0, :], v4[0, :]))


class testHopf(unittest.TestCase):

    def testProject(self):
        
        pass


if __name__ == '__main__':
    unittest.main()

