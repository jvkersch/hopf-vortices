import sys
import numpy as np
import scipy.optimize as so

from generic_integrator import GenericIntegrator
from ..vortices.continuous_vortex_system import vortex_rhs
from ..lie_algebras.lie_algebra import cayley
from ..util.vectors import row_product
from ..lie_algebras.su2_geometry import (cayley_klein, apply_2by2, hopf, 
                                         inverse_hopf)
from ..vortices.continuous_vortex_system import scaled_gradient_hamiltonian


class VortexIntegrator:

    def __init__(self, gamma, sigma=0.0, h=1e-1, 
                 verbose=False, callback=None):
        """
        callback -- function to call after each iteration of the integrator
        for postprocessing.

        """

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

        self.callback = callback


    def iteration_direct(self, b, psi0, x0):
        """Return update for `b` via direct fixed-point equation."""

        a = self.half_time*b
        U = cayley_klein(a)
        psi1 = apply_2by2(U, psi0)
        x1 = hopf(psi1)

        gradH = scaled_gradient_hamiltonian(self.gamma, (x0+x1)/2., self.sigma)
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
    
        gradH = scaled_gradient_hamiltonian(self.gamma, (x0+x1)/2., self.sigma)
        dot   = np.sum(x1*gradH, axis=1)

        return 1./4*np.cross(np.cross(gradH, x1, axis=1) + row_product(dot, a), 
                             x1, axis=1)


    def residue_adjoint(self, b, psi0, x0):
        """Residue for adjoint iteration."""
        return b - self.iteration_adjoint(b, psi0, x0)


    def integrate(self, X0, tmax=50., numpoints=100, full_output=False):

        num_inner = int(round(tmax/(self.h*numpoints)))
        t = 0

        vortices = np.zeros((numpoints, ) + X0.shape)
        times = np.zeros(numpoints)

        psi0 = inverse_hopf(X0)

        if self.verbose:
            print >> sys.stderr, "Entering integration loop"

        for k in xrange(0, numpoints):
            print >> sys.stderr, '.',
            for _ in xrange(0, num_inner):
                psi0, X0 = self.do_one_step(t, psi0, X0)
                t += 2*self.half_time

            # Save output
            vortices[k, :, :] = X0
            times[k] = t


        print >> sys.stderr, '\n'
        return vortices, times


    def do_one_step(self, t, psi0, x0):

        # Apply direct method
        f = lambda y: self.residue_direct(y, psi0, x0)
        b0 = so.newton_krylov(f, self.b, f_tol=1e-14)
        U0 = cayley_klein(self.half_time*b0)
        psi1 = apply_2by2(U0, psi0); x1 = hopf(psi1)

        # Apply adjoint method
        f = lambda y: self.residue_adjoint(y, psi1, x1)
        b1 = so.newton_krylov(f, b0, f_tol=1e-14)
        U1 = cayley_klein(self.half_time*b1)
        psi2 = apply_2by2(U1, psi1); x2 = hopf(psi2)

        # Save b2 for next iteration
        self.b = b2

        # Run callbacks
        if self.callback is not None:
            self.callback(x0, psi0, b0, x1, psi1, b1, x2, psi2)

        return psi2, x2




