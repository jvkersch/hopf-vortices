import numpy as np
import unittest

from hopf.vortices.continuous_vortex_system import vortex_hamiltonian
from hopf.vortices.vortices_s3 import vortex_hamiltonian_S3
from hopf.lie_algebras.su2_geometry import hopf, pauli
from hopf.util.vectors import row_product

class HamiltonianTest(unittest.TestCase):

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


    def test_hamiltonian_S2_S3(self):
        """
        Compare the value of the Hamiltonian function on S2 with the one 
        on S3.

        """

        H1 = vortex_hamiltonian_S3(self.gamma, self.phi, self.sigma)
        H2 = vortex_hamiltonian(self.gamma, self.x, self.sigma)

        self.assertTrue( abs(H1-H2) < 1e-12 )


if __name__ == '__main__':
    unittest.main()
