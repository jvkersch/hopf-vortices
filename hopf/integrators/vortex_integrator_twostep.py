import sys
import numpy as np
import scipy.optimize as so

from generic_integrator import GenericIntegrator
from ..lie_algebras.lie_algebra import cayley
from ..util.vectors import row_product
from ..lie_algebras.su2_geometry import (cayley_klein, apply_2by2, hopf, 
                                         inverse_hopf, pauli)
from ..vortices.vortices_s3 import scaled_gradient_hamiltonian_S3


class VortexIntegratorTwostep:

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

    def residual_direct(self, phi0, phi1):
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

        return res

    def residual_adjoint(self, phi0, phi1):
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

        return res


    def residual_adjoint_su2(self, psi0, b0):
        """
        Computes the residual of the adjoint equation, expressed as a function
        of the initial point `psi0` and the update element `b0`.

        """

        U0 = cayley_klein(self.half_time*b0)
        psi1 = apply_2by2(U0, psi0)
        
        return self.residual_adjoint(psi0, psi1)


    def residual_direct_su2(self, psi0, b0):
        """
        Computes the residual of the direct equation, expressed as a function
        of the initial point `psi0` and the update element `b0`.

        """

        U0 = cayley_klein(self.half_time*b0)
        psi1 = apply_2by2(U0, psi0)
        
        return self.residual_direct(psi0, psi1)
        

    def bootstrap(self, psi0):
        """
        Given an initial point `psi0`, finds the next point `psi` by 
        solving the adjoint equation.

        """
        f = lambda b: self.residual_adjoint_su2(psi0, b)
        self.b = so.newton_krylov(f, self.b, f_tol=1e-14)
        U0 = cayley_klein(self.half_time*self.b)
        psi1 = apply_2by2(U0, psi0)

        return psi1


    def check_full_equations(self, phi0, phi1, phi2):
        """
        Helper method to check whether a triple of points satisfies the 
        full two-point discrete Euler-Lagrange equations.

        """

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


    def compute_momentum_map(self, phi0, phi1):
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


    def integrate(self, X0, tmax=50., numpoints=100, full_output=False):

        num_inner = int(round(tmax/(self.h*numpoints)))
        t = 0

        momentum = np.empty((numpoints, 3))
        vortices = np.empty((numpoints, ) + X0.shape)
        times = np.empty(numpoints)

        psi0 = inverse_hopf(X0)
        X0 = hopf(psi0)

        psi1 = self.bootstrap(psi0)
        X1 = hopf(psi1)

        t += self.half_time

        if self.verbose:
            print >> sys.stderr, "Entering integration loop"

        for k in xrange(0, numpoints):
            print >> sys.stderr, '.',
            for _ in xrange(0, num_inner):
                psi2, X2, m = self.do_one_step(t, psi0, X0, psi1, X1)
                t += self.half_time

                psi0 = psi1; psi1 = psi2;
                X0 = X1; X1 = X2;

            # Save output
            vortices[k, :, :] = X2
            times[k] = t

            if self.compute_momentum:
                momentum[k, :] = m


        print >> sys.stderr, '\n'

        if self.compute_momentum:
            return vortices, times, momentum
        else:
            return vortices, times


    def do_one_step(self, t, psi0, x0, psi1, x1):
        """
        Take one integration step.
        
        """

        slack = self.residual_adjoint(psi0, psi1)

        f = lambda b: self.residual_direct_su2(psi1, b) + slack
        self.b = so.newton_krylov(f, self.b, f_tol=1e-14)
        U1 = cayley_klein(self.half_time*self.b)
        psi2 = apply_2by2(U1, psi1); x2 = hopf(psi2)

        # print self.check_full_equations(psi0, psi1, psi2)

        # Compute momentum map, if needed
        m = None
        if self.compute_momentum:
            m = self.compute_momentum_map(psi1, psi2)

        return psi2, x2, m





