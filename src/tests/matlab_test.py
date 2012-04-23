"""
Test methods against results previously obtained with a cruder 
Matlab version of the code.
"""

import sys
sys.path.append('..')

import unittest
import numpy as np
from matlab_io import *
from vectors import row_product, is_float_equal

test_suite = 'medium_system.mat'

class testMatlab(unittest.TestCase):

    def setUp(self):
        [b, x0, gamma, grad, psi0, h, res_adjoint, res_direct, sigma] = \
            load_variables_from_matfile(test_suite, 
                                        ['B', 'X0', 'gamma_row', 'grad',
                                         'psi0', 'h', 'res_adjoint', 
                                         'res_direct', 'sigma'])

        self.b = b
        self.x0 = x0
        self.gamma = gamma[0]
        self.grad = grad
        self.psi0 = psi0
        self.h = h
        self.res_adjoint = res_adjoint
        self.res_direct = res_direct
        self.sigma = sigma[0][0]


    def testHamiltonian(self):
        from continuous_vortex_system import scaled_gradient_hamiltonian
        scaled_grad = scaled_gradient_hamiltonian(self.gamma, self.x0, 
                                                  self.sigma)
        
        grad2 = row_product(self.gamma, scaled_grad) 
        self.assertTrue(is_float_equal(grad2, self.grad, tol=1e-14))


    def testResidueDirect(self):
        from vortex_integrator import iteration_direct, VortexSystem

        v = VortexSystem()
        v.h = self.h 
        v.gamma = self.gamma
        v.sigma = self.sigma

        r = iteration_direct(self.b, self.x0, v)
        r = 2*row_product(self.gamma, self.b - r)
        self.assertTrue(is_float_equal(r, self.res_direct, tol=1e-14))


    def testResidueAdjoint(self):
        from vortex_integrator import iteration_adjoint, VortexSystem

        v = VortexSystem()
        v.h = self.h 
        v.gamma = self.gamma
        v.sigma = self.sigma

        r = iteration_adjoint(self.b, self.x0, v)
        r = 2*row_product(self.gamma, self.b - r)

        self.assertTrue(is_float_equal(r, self.res_adjoint, tol=1e-14))



if __name__ == '__main__':
    unittest.main()

        



