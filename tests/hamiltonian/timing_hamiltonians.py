import numpy as np

from hopf.vortices.vortices_S2 import gradient_S2, gradient_S2_slow
from hopf.vortices.vortices_S3 import gradient_S3

from hopf.lie_algebras.su2_geometry import hopf, pauli
from hopf.util.vectors import row_product

# Constants
N = 5
sigma = .45

# Random initial conditions
phi = np.random.rand(N, 2) + 1j*np.random.rand(N, 2)
gamma = np.random.rand(N)

# Normalize
norms = np.sum(phi.conj()*phi, axis=1)**.5
phi = row_product(1./norms, phi)

# Compute projected point
x = hopf(phi)

# For optimized gradient
DH_S2 = np.zeros(x.shape)
DH_S3 = np.zeros(phi.shape, dtype=np.complex)

def compute_optimized_gradient_hamiltonian_S3():
    gradient_S3(DH_S3, gamma, phi, sigma)

def compute_optimized_gradient_hamiltonian_S2():
    gradient_S2(DH_S2, gamma, x, sigma)

def compute_gradient_hamiltonian_S2():
    gradient_S2_slow(gamma, x, sigma)


if __name__ == '__main__':
    import timeit

    print("Gradient of Hamiltonian on S2: %f" % timeit.timeit(
            "compute_gradient_hamiltonian_S2()", 
            setup="from __main__ import compute_gradient_hamiltonian_S2",
            number=5))

    print("Optimized gradient of Hamiltonian on S2: %f" % timeit.timeit(
            "compute_optimized_gradient_hamiltonian_S2()", 
            setup="from __main__ import compute_optimized_gradient_hamiltonian_S2",
            number=5))

    print("Optimized gradient of Hamiltonian on S3: %f" % timeit.timeit(
            "compute_optimized_gradient_hamiltonian_S3()", 
            setup="from __main__ import compute_optimized_gradient_hamiltonian_S3",
            number=5))
