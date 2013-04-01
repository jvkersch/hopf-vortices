import numpy as np

from ..lie_algebras.su2_geometry import pauli
#from ..vortices.continuous_vortex_system import vortex_hamiltonian


def hermitian_product(phi1, phi2):
    """
    Compute the Hermitian inner product of two complex vectors.

    """
    return np.dot( phi1.conj(), phi2 )


def vortex_hamiltonian_S3(gamma, phi, sigma):
    """
    Value of the point vortex Hamiltonian evaluated at a point of the 3-sphere.

    """

    H = 0
    N = phi.shape[0]
    for k in xrange(0, N):
        for l in xrange(k+1, N):
            p = hermitian_product(phi[k, :], phi[l, :])
            H -= ( gamma[k]*gamma[l]/(4*np.pi) * 
                   np.log(2*sigma**2 + 4*(1 - p*p.conj())) ).real

    return H


def scaled_gradient_hamiltonian_S3(gamma, phi, sigma):
    """
    Gradient of the point vortex Hamiltonian on the three-sphere S3, where
    each entry has been scaled by the corresponding vortex strength.

    Note: the derivatives here are with respect to the conjugate variables.

    """
    N = phi.shape[0]
    DH = np.zeros((N, 2), dtype=np.complex)

    for k in xrange(0, N):
        for l in xrange(0, N):
            if l == k: continue
            p = hermitian_product(phi[l, :], phi[k, :])
            DH[k, :] += (gamma[l]/np.pi*p*phi[l, :] / 
                         (2*sigma**2 + 4*(1 - p*p.conj())))

    return DH


def scaled_gradient_hamiltonian_S3_finite_differences(gamma, phi, sigma):
    """
    Compute the gradient of the Hamiltonian on S3 with finite differences.

    """
    N = phi.shape[0]
    DH = np.zeros((N, 2), dtype=np.complex)

    e1 = np.array([1. , 0])
    e2 = np.array([1.j, 0])
    e3 = np.array([0, 1. ])
    e4 = np.array([0, 1.j])

    eps = 1e-5

    for k in xrange(0, N):
        for n, (v, w) in enumerate(((e1, e2), (e3, e4))):

            diff = np.zeros((N, 2), dtype=np.complex)

            diff[k, :] = v
            re = ( (vortex_hamiltonian_S3(gamma, phi + eps*diff, sigma) -
                    vortex_hamiltonian_S3(gamma, phi - eps*diff, sigma) )/
                   (2*eps) )

            diff[k, :] = w
            im = ( (vortex_hamiltonian_S3(gamma, phi + eps*diff, sigma) -
                    vortex_hamiltonian_S3(gamma, phi - eps*diff, sigma) )/
                   (2*eps) )

            DH[k, n] = (re + 1.j*im)/2 # Complex derivative

    return DH



def projection(phi, d_phi):
    """
    Compute the real part of i d_phi sigma phi, where sigma are the Pauli
    matrices.

    """

    N = phi.shape[0]
    vec = np.zeros((N, 3))

    for i in xrange(0, 3):
        for k in xrange(0, N):
            entry = np.dot(d_phi[k, :], np.dot(pauli[:, :, i], phi[k, :]))
            vec[k, i] = (1j*entry).real

    return vec
        
