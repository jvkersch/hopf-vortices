import sys
import numpy as np
import scipy.optimize as so

from generic_integrator import GenericIntegrator
from ..vortices.continuous_vortex_system import vortex_rhs

from ..util.vectors import row_product
from ..util.array_solver import FSolveArray
from ..lie_algebras.su2_geometry import (cayley_klein, apply_2by2, 
                                       hopf, inverse_hopf)
from ..vortices.continuous_vortex_system import scaled_gradient_hamiltonian
from .diagnostics import BroydenDiagnostics


class VortexIntegrator_mu:

    def __init__(self, gamma, sigma=0.0, h=1e-1, 
                 verbose=False, diagnostics=False):

        self.gamma = np.array(gamma)
        self.sigma = sigma
        self.N = self.gamma.size

        self.b = np.zeros((self.N, 3), dtype=np.double)

        self.half_time = h
        self.h = 2*h
        self.verbose = verbose

        self.diagnostics = diagnostics
        self.diagnostics_logger = BroydenDiagnostics()

    def residue_mu_adjoint(self, b, psi0, x0):
        """
        Calculate Lie-algebra element with nonvanishing parallel component.
        
        """
        a = self.half_time*b
        U = cayley_klein(a)
        psi1 = apply_2by2(U, psi0)
        x1 = hopf(psi1)

        gradH = scaled_gradient_hamiltonian(self.gamma, (x0+x1)/2.0, self.sigma)
        dot = np.sum(x1*gradH, axis=1)
        mu = -self.half_time/4.0*(
            dot - np.sum(a*np.cross(gradH, x1, axis=1), axis=1))

        return a + 1.0/4*self.half_time*np.cross(
            np.cross(x1, gradH, axis=1) - row_product(dot, a) + 
            row_product(mu, gradH), x1, axis=1) - row_product(mu, x1)

    def residue_mu_direct(self, b, psi0, x0):
        """
        Calculate Lie-algebra element with nonvanishing parallel component.
        
        """
        a = self.half_time*b
        U = cayley_klein(a)
        psi1 = apply_2by2(U, psi0)
        x1 = hopf(psi1)

        gradH = scaled_gradient_hamiltonian(self.gamma, (x0+x1)/2.0, self.sigma)
        dot = np.sum(x0*gradH, axis=1)
        mu = -self.half_time/4.0*(
            dot + np.sum(a * np.cross(gradH, x0, axis=1), axis=1))

        return a + 1.0/4*self.half_time*np.cross(
            np.cross(x0, gradH, axis=1) + row_product(dot, a) - 
            row_product(mu, gradH), x0, axis=1) - row_product(mu, x0)

    def integrate(self, X0, tmax = 50.0, numpoints=100, full_output=False):

        num_inner = int(round(tmax/(self.h*numpoints)))
        t = 0

        vortices = np.zeros((numpoints,) + X0.shape)
        times = np.zeros(numpoints)

        psi0 = inverse_hopf(X0)

        if self.verbose:
            print >> sys.stderr, 'Entering integration loop'

        for k in xrange(0, numpoints):
            print >> sys.stderr, '.',
            for _ in xrange(0, num_inner):
                psi0, X0 = self.do_one_step(t, psi0, X0)
                t += 2*self.half_time

            vortices[k, :, :] = X0
            times[k] = t

        print >> sys.stderr, '\n'
        return vortices, times

    def do_one_step(self, t, psi0, x0):

        callback = None
        if self.diagnostics:
            callback = self.diagnostics_logger

        f = lambda y: self.residue_mu_direct(y, psi0, x0)
        self.b = so.newton_krylov(f, self.b, f_tol=1e-14, 
                                  callback=callback, verbose=False)
        U = cayley_klein(self.half_time*self.b)
        psi0 = apply_2by2(U, psi0)
        x0 = hopf(psi0)

        self.diagnostics_logger.store()

        f = lambda y: self.residue_mu_adjoint(y, psi0, x0)
        self.b = so.newton_krylov(f, self.b, f_tol=1e-14, 
                                  callback=callback, verbose=False)
        U = cayley_klein(self.half_time * self.b)
        psi0 = apply_2by2(U, psi0)
        x0 = hopf(psi0)

        self.diagnostics_logger.store()

        return psi0, x0
