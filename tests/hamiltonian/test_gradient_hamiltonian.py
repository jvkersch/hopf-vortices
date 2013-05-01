import numpy as np
import unittest

from hopf.vortices.vortices_S2 import gradient_S2, gradient_S2_slow
from hopf.vortices.vortices_S3 import gradient_S3, gradient_S3_fd

from hopf.lie_algebras.su2_geometry import hopf, pauli
from hopf.util.vectors import row_product


def gradient_S3_pullback(gamma, psi, sigma):
    """
    Compute the gradient on S3 by first calculating the gradient on S2
    and pulling back the result to S3.

    """

    x = hopf(psi)
    grad_S2 = gradient_S2_slow(gamma, x, sigma)
    
    return np.einsum('ij, abj, ib -> ia', grad_S2, pauli, psi)


class GradientHamiltonianTest(unittest.TestCase):

    N = 40  # Number of vortices
    sigma = .45 # Regularization

    def setUp(self):
        # Random initial conditions
        self.gamma = np.random.rand(self.N)        
        
        phi = np.random.rand(self.N, 2) + 1j*np.random.rand(self.N, 2)

        # Normalize phi
        norms = np.sum(phi.conj()*phi, axis=1)**.5
        phi = row_product(1./norms, phi)

        # Perturb away from the unit sphere somewhat
        phi *= 1.01

        self.phi = phi

        # Compute projected point
        self.x = hopf(phi)

    def test_compare_S3_optimized(self):
        """
        Compare gradient implemented with finite differences with the 
        optimized cython gradient.

        """

        # Pre-allocate buffer
        DH3a = np.zeros(self.phi.shape, dtype=np.complex)

        gradient_S3(DH3a, self.gamma, self.phi, self.sigma)
        DH3b = gradient_S3_fd(self.gamma, self.phi, self.sigma)

        self.assertTrue( np.allclose(DH3a, DH3b) )

    def test_compare_S2_optimized(self):
        """
        Compare different implementations of gradient on S2.

        """

        # Pre-allocate buffer
        DH2a = np.zeros(self.x.shape)

        gradient_S2(DH2a, self.gamma, self.x, self.sigma)
        DH2b = gradient_S2_slow(self.gamma, self.x, self.sigma)

        self.assertTrue( np.allclose(DH2a, DH2b, atol=1e-12) )


    def test_compare_S2_S3(self):
        """
        Compare the gradient of the Hamiltonian on S2 with the gradient on 
        S3.

        """


        # Pre-allocate buffer
        DH3a = np.zeros(self.phi.shape, dtype=np.complex)
        gradient_S3(DH3a, self.gamma, self.phi, self.sigma)
        DH3b = gradient_S3_pullback(self.gamma, self.phi, self.sigma)

        self.assertTrue( np.allclose(DH3a, DH3b, atol=1e-12) )


if __name__ == '__main__':
    unittest.main()
