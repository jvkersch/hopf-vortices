import sys
import numpy as np
import scipy.optimize as so

from generic_integrator import GenericIntegrator
from ..vortices.continuous_vortex_system import vortex_rhs
from ..lie_algebras.lie_algebra import cayley
from ..util.vectors import row_product
from ..lie_algebras.su2_geometry import (cayley_klein, apply_2by2, hopf, 
                                         inverse_hopf, pauli)
from ..vortices.continuous_vortex_system import scaled_gradient_hamiltonian
from ..vortices.vortices_s3 import scaled_gradient_hamiltonian_S3


"""
TODO: 

* get rid of extraneous imports 

* get rid of separate residue/iteration methods 

"""

class VortexIntegrator:

    def __init__(self, gamma, sigma=0.0, h=1e-1, 
                 verbose=False, compute_momentum=False):

        self.gamma = np.array(gamma)
        self.sigma = sigma
        self.N = self.gamma.size

        # Initial choice for update element
        self.b = np.zeros((self.N, 3), dtype=np.double)

        # Initialize base class
        # Note that since this is a composition method, each iteration 
        # will take two steps of size h, resulting in an overall time 
        # step per iteration of 2*h
        self.half_time = h
        self.h = 2*h
        self.verbose = verbose

        self.compute_momentum = compute_momentum

    def check_direct_equation_S3(self, phi0, phi1):
        """
        Compute the residual of the direct equation.

        """

        gradH01 = scaled_gradient_hamiltonian_S3(self.gamma, (phi0+phi1)/2,
                                                 self.sigma)

        N = len(self.gamma)
        res = np.empty((N, 3))

        for k in xrange(0, N):
            f = -1.j*(phi1[k, :] - phi0[k, :]) + \
                self.half_time/2*(gradH01[k, :])
            for alpha in xrange(0, 3):
                projector = np.dot(phi0[k, :].conj(), 1.j*pauli[:, :, alpha])
                res[k, alpha] = np.dot(projector, f).real

        return np.max(np.max(np.abs(res)))


    def check_adjoint_equation_S3(self, phi0, phi1):
        """
        Compute the residual of the adjoint equation.

        """
        gradH01 = scaled_gradient_hamiltonian_S3(self.gamma, (phi0+phi1)/2,
                                                 self.sigma)

        N = len(self.gamma)
        res = np.empty((N, 3))

        for k in xrange(0, N):
            f = -1.j*(phi1[k, :] - phi0[k, :]) + \
                self.half_time/2*(gradH01[k, :])
            for alpha in xrange(0, 3):
                projector = np.dot(phi1[k, :].conj(), 1.j*pauli[:, :, alpha])
                res[k, alpha] = np.dot(projector, f).real

        return np.max(np.max(np.abs(res)))



    def check_equations_S3(self, phi0, phi1, phi2):

        gradH01 = scaled_gradient_hamiltonian_S3(self.gamma, (phi0+phi1)/2,
                                                 self.sigma)
        
        gradH12 = scaled_gradient_hamiltonian_S3(self.gamma, (phi1+phi2)/2,
                                                 self.sigma)

        N = len(self.gamma)
        res = np.empty((N, 3))

        for k in xrange(0, N):
            f = -1.j*(phi2[k, :] - phi0[k, :]) + \
                self.half_time/2*(gradH01[k, :] + gradH12[k, :])
            for alpha in xrange(0, 3):
                projector = np.dot(phi1[k, :].conj(), 1.j*pauli[:, :, alpha])
                res[k, alpha] = np.dot(projector, f).real

        return np.max(np.max(np.abs(res)))


    def compute_momentum_map(self, x0, a0, x1): 
        """
        TODO: this method is still buggy. At the very least, Hamiltonian
        needs to be evaluated at pi( (phi0 + phi1)/2 ) rather than 
        (x0 + x1)/2 as is now the case.

        """
        gradH = scaled_gradient_hamiltonian(self.gamma, (x0+x1)/2., self.sigma)
        norm_a0 = np.sum(a0**2, axis=1) 

        term1 = row_product(2./(1 + norm_a0), np.cross(a0, x0, axis=1) + 
                            row_product(norm_a0, x0)) - x0
        term2 = row_product(1./(1 + norm_a0), np.cross(x0, gradH, axis=1) 
                            - np.cross(np.cross(a0, x0, axis=1), gradH, axis=1))

        return np.sum(row_product(self.gamma, term1 - 
                                  self.half_time/2*term2), axis=0)


    def compute_momentum_map_S3(self, phi0, phi1):
        """
        Compute the vortex momentum map using geometric quantities defined
        directly on S3.

        """
        gradH = scaled_gradient_hamiltonian_S3(self.gamma, (phi0+phi1)/2,
                                               self.sigma).conj()

	N = len(self.gamma)
        J = np.zeros(3)

        for alpha in xrange(0, 3):
		for k in xrange(0, N):
            		P = 1j*np.dot(pauli[:, :, alpha], phi0[k, :])

            		term1 = 1j*phi1[k,:].conj()
            		term2 = self.half_time/2*gradH[k, :]

		        J[alpha] += self.gamma[k]*np.dot(term1 + term2, P).real

        return J


    def iteration_direct(self, b, psi0, x0):
        """Return update for `b` via direct fixed-point equation."""

        a = self.half_time*b
        U = cayley_klein(a)
        psi1 = apply_2by2(U, psi0)
        x1 = hopf(psi1)

        x01 = hopf((psi0+psi1)/2)

        gradH = scaled_gradient_hamiltonian(self.gamma, x01, self.sigma)
        dot   = np.sum(x0*gradH, axis=1)

        return 1./4*np.cross(np.cross(gradH, x0, axis=1) - row_product(dot, a), 
                             x0, axis=1)


    def residue_direct(self, b, psi0, x0):
        """Residue for direct iteration."""
        return b - self.iteration_direct(b, psi0, x0)


    def iteration_adjoint(self, b, psi0, x0):
        """Return update for `b` via adjoint fixed-point equation."""
    
        a = self.half_time*b
        U = cayley_klein(a)
        psi1 = apply_2by2(U, psi0)
        x1 = hopf(psi1)

        x01 = hopf((psi0+psi1)/2)
    
        gradH = scaled_gradient_hamiltonian(self.gamma, x01, self.sigma)
        dot   = np.sum(x1*gradH, axis=1)

        return 1./4*np.cross(np.cross(gradH, x1, axis=1) + row_product(dot, a), 
                             x1, axis=1)


    def residue_adjoint(self, b, psi0, x0):
        """Residue for adjoint iteration."""
        return b - self.iteration_adjoint(b, psi0, x0)


    def integrate(self, X0, tmax=50., numpoints=100, full_output=False):

        num_inner = int(round(tmax/(self.h*numpoints)))
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
                psi0, X0, m = self.do_one_step(t, psi0, X0)
                t += 2*self.half_time

            # Save output
            vortices[k, :, :] = X0
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

        # Apply adjoint method
        f = lambda y: self.residue_adjoint(y, psi0, x0)
        b0 = so.newton_krylov(f, self.b, f_tol=1e-14)
        U0 = cayley_klein(self.half_time*b0)
        psi1 = apply_2by2(U0, psi0); x1 = hopf(psi1)

        # Apply direct method
        f = lambda y: self.residue_direct(y, psi1, x1)
        b1 = so.newton_krylov(f, b0, f_tol=1e-14)
        U1 = cayley_klein(self.half_time*b1)
        psi2 = apply_2by2(U1, psi1); x2 = hopf(psi2)

        self.b = b1


        # Compute momentum map, if needed
        m = None
        if self.compute_momentum:
            m = self.compute_momentum_map_S3(psi0, psi1)

        return psi2, x2, m





