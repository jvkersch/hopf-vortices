import numpy as np
import unittest

from hopf.vortices.continuous_vortex_system import scaled_gradient_hamiltonian
from hopf.vortices.vortices_s3 import scaled_gradient_hamiltonian_S3, scaled_gradient_hamiltonian_S3_fd
from hopf.vortices.continuous_vortex_system_S3 import optimized_scaled_gradient_hamiltonian_S3

from hopf.lie_algebras.su2_geometry import hopf, pauli
from hopf.util.vectors import row_product


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

    def test_compare_S3_finite_differences(self):
        """
        Compare analytic expression for the gradient of Hamiltonian 
        on S3 with the gradient computed by finite differences.

        """

        DH3_an = scaled_gradient_hamiltonian_S3(self.gamma, self.phi, 
                                                self.sigma)
        DH3_fd = scaled_gradient_hamiltonian_S3_fd(self.gamma, self.phi, 
                                                   self.sigma)

        self.assertTrue( np.allclose(DH3_an, DH3_fd) )

    def test_compare_S3_optimized(self):
        """
        Compare analytic expression for gradient implemented in pure
        Python with optimized cython version.

        """

        # Pre-allocate buffer
        DH3_cy = np.zeros(self.phi.shape, dtype=np.complex)

        optimized_scaled_gradient_hamiltonian_S3(DH3_cy, self.gamma, self.phi, 
                                                 self.sigma)
        DH3_an = scaled_gradient_hamiltonian_S3(self.gamma, self.phi, 
                                                self.sigma)

        self.assertTrue( np.allclose(DH3_cy, DH3_an, atol=1e-12) )

    def test_compare_S2_S3(self):
        """
        Compare the gradient of the Hamiltonian on S2 with the gradient on 
        S3.

        """

        DH3_an = scaled_gradient_hamiltonian_S3(self.gamma, self.phi, 
                                                self.sigma)

        DH2_an = scaled_gradient_hamiltonian(self.gamma, self.x, self.sigma)

        # Pull back gradient on S2 to S3
        pullback = np.zeros((self.N, 2), dtype=np.complex)
        for k in xrange(0, self.N):
            for i in xrange(0, 3):
                pullback[k, :] += DH2_an[k, i]*np.dot(pauli[:, :, i], 
                                                      self.phi[k, :])

        self.assertTrue( np.allclose(pullback, DH3_an, atol=1e-12) )


if __name__ == '__main__':
    unittest.main()
