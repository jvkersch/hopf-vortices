import sys
import numpy as np
import scipy.optimize as so

from generic_integrator import GenericIntegrator
from ..lie_algebras.lie_algebra import cayley
from ..util.vectors import row_product
from ..lie_algebras.su2_geometry import (cayley_klein, apply_2by2, hopf, 
                                         inverse_hopf, pauli)

from ..vortices.vortices_S2 import gradient_S2
from ..vortices.vortices_S3 import gradient_S3

import ipdb


def complex_fsolve(fun, psi0, **kwargs):
    """
    Wrapper to solve complex-valued nonlinear equations using newton_krylov.

    (Workaround for a bug in scipy < 0.11)

    """
    m, n = psi0.shape
    def real_fun(x):
        out = fun(x[:, :n] + 1j*x[:, n:])
        return np.hstack([out.real, out.imag])
    c_out = so.newton_krylov(real_fun, np.hstack([psi0.real, psi0.imag]), 
                             **kwargs)
    return c_out[:, :n] + 1j*c_out[:, n:]


class VortexIntegratorMidpoint:

    def __init__(self, gamma, sigma=0.0, h=1e-1, 
                 verbose=False, compute_momentum=False):

        self.gamma = np.array(gamma)
        self.sigma = sigma
        self.N = self.gamma.size

        self.h = h
        self.verbose = verbose
        self.compute_momentum = compute_momentum


    def residual_midpoint_eqn(self, phi0, phi1, pullback=False):
        """
        Compute the residual of the midpoint equation. 
        Needed for nonlinear solver below.

        An alternative way to compute the gradient is by using

        >>> x01 = hopf(phi01)
        >>> gradH01_S2 = scaled_gradient_hamiltonian(self.gamma, x01,
                                                 self.sigma)
        >>> gradH01_S3a = np.einsum('ij, abj, ib -> ia',
                                    gradH01_S2, pauli, phi01)


        """
        phi01 = (phi0 + phi1) / 2

        gradH01_S3 = np.empty(phi01.shape, dtype=np.complex)
        gradient_S3(gradH01_S3, self.gamma, phi01, self.sigma)

        return -1.j*(phi1 - phi0) + self.h/2*gradH01_S3


    def check_full_equations(self, phi0, phi1, phi2):
        """
        Helper method to check whether a triple of points satisfies the 
        full two-point discrete Euler-Lagrange equations.

        TODO: This is not the most efficient implementation possible, so
        use for debugging purposes only.

        """
        gradH01 = np.empty(phi01.shape, dtype=np.complex)
        gradH12 = np.empty(phi01.shape, dtype=np.complex)
        scaled_gradient_S3(gradH01, self.gamma, (phi0 + phi1)/2, self.sigma)
        scaled_gradient_S3(gradH12, self.gamma, (phi1 + phi2)/2, self.sigma)


        N = len(self.gamma)
        res = np.empty((N, 3))

        for k in xrange(0, N):
            f = -1.j*(phi2[k, :] - phi0[k, :]) + \
                self.h/2*(gradH01[k, :] + gradH12[k, :])
            for alpha in xrange(0, 3):
                projector = np.dot(phi1[k, :].conj(), 1.j*pauli[:, :, alpha])
                res[k, alpha] = np.dot(projector, f).real

        return np.max(np.max(np.abs(res)))


    def compute_momentum_map(self, phi0, phi1):
        """
        Compute the vortex momentum map using geometric quantities defined
        directly on S3.

        """
        gradH = np.empty(phi0.shape, dtype=np.complex)
        gradient_S3(gradH, self.gamma, (phi0 + phi1)/2, self.sigma)

        gradH = gradH.conj()

	N = len(self.gamma)
        J = np.zeros(3)

        for alpha in xrange(0, 3):
            for k in xrange(0, N):
                P = 1j*np.dot(pauli[:, :, alpha], phi0[k, :])
                
                term1 = 1j*phi1[k,:].conj()
                term2 = self.h/2*gradH[k, :]

                J[alpha] += self.gamma[k]*np.dot(term1 + term2, P).real

        return J


    def integrate(self, X0, tmax=50., numpoints=100, full_output=False):

        num_inner = int(round(tmax/(self.h*numpoints))) # ...
        t = 0

        momentum = np.empty((numpoints, 3))
        vortices = np.empty((numpoints, ) + X0.shape)
        times = np.empty(numpoints)

        psi0 = inverse_hopf(X0)
        X0 = hopf(psi0)

        if self.verbose:
            print >> sys.stderr, "Entering integration loop"

        for k in xrange(0, numpoints):
            print >> sys.stderr, '.',
            for _ in xrange(0, num_inner):
                psi1, X1, m = self.do_one_step(t, psi0, X0)
                t += self.h

                psi0 = psi1; 
                X0 = X1; 

            # Save output
            vortices[k, :, :] = X1
            times[k] = t

            if self.compute_momentum:
                momentum[k, :] = m

        print >> sys.stderr, '\n'

        if self.compute_momentum:
            return vortices, times, momentum
        else:
            return vortices, times


    def do_one_step(self, t, psi0, x0):
        """
        Take one integration step.
        
        """

        f = lambda psi1: self.residual_midpoint_eqn(psi0, psi1)
        psi1 = complex_fsolve(f, psi0, f_tol=1e-14, verbose=False)
        # psi1 = so.newton_krylov(f, psi0, f_tol=1e-14, verbose=False)
        x1 = hopf(psi1)

        # Compute momentum map, if needed
        m = None
        if self.compute_momentum:
            m = self.compute_momentum_map(psi0, psi1)

        return psi1, x1, m
