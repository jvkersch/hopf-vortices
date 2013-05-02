import numpy as np

from hopf.lie_algebras.su2_geometry import hopf
from hopf.util.vectors import row_product
from hopf.util.matlab_io import save_variables


def create_random_initial_conditions(N):
    """
    Create an ensemble of `N` randomly chosen vortices and vortex strengths.

    """
    phi = np.random.rand(N, 2) + 1j*np.random.rand(N, 2)
    gamma = np.random.rand(N)
    
    # Normalize
    norms = np.sum(phi.conj()*phi, axis=1)**.5
    phi = row_product(1./norms, phi)

    return phi, gamma

if __name__ == '__main__':
    print "Creating 40 random vortices of strength 1/8..."

    phi, _ = create_random_initial_conditions(40)
    gamma = 1./8 * np.ones(40)
    sigma = 0.1
    x = hopf(phi)

    save_variables('random40.mat', {'gamma': gamma, 'X0': x, 'sigma': sigma})
