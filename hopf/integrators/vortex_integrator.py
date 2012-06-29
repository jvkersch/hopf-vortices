import sys
import numpy as np
import scipy.optimize as so
#from math import ceil

from generic_integrator import GenericIntegrator
from ..vortices.continuous_vortex_system import vortex_rhs
from ..lie_algebras.lie_algebra import cayley

# TODO Method docstrings are horrible
# TODO Can we bring some clarity into the morass of different optimization methods?

from ..util.vectors import row_product
from ..util.array_solver import FSolveArray
from ..lie_algebras.su2_geometry import (cayley_klein, apply_2by2, hopf, 
                                         inverse_hopf)
from ..vortices.continuous_vortex_system import scaled_gradient_hamiltonian

from .diagnostics import BroydenDiagnostics


class VortexIntegrator:

    def __init__(self, gamma, sigma=0.0, h=1e-1, 
                 verbose=False, diagnostics=False):

        self.gamma = np.array(gamma)
        self.sigma = sigma
        self.N = self.gamma.size

        # Set up nonlinear solvers
        #size = (self.N, 3)
        #self.solver_direct  = FSolveArray(self.residue_direct, size=size)
        #self.solver_adjoint = FSolveArray(self.residue_adjoint, size=size)

        # Diagnostics
        #self.res = np.zeros(self.N, dtype=np.double)

        # Initial choice for update element
        self.b = np.zeros((self.N, 3), dtype=np.double)

        # Initialize base class
        # Note that since this is a composition method, each iteration 
        # will take two steps of size h, resulting in an overall time 
        # step per iteration of 2*h
        self.half_time = h
        self.h = 2*h
        self.verbose = verbose

        # Keep track of nonlinear convergence
        self.diagnostics = diagnostics
        self.diagnostics_logger = BroydenDiagnostics()


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


    def do_one_step_fixedpoint(self, t, psi0, x0):

        f = lambda y: self.iteration_direct(y, psi0, x0)
        self.b = so.fixed_point(f, self.b, xtol=1e-12)
        U = cayley_klein(self.half_time*self.b)
        psi0 = apply_2by2(U, psi0); x0 = hopf(psi0)        

        f = lambda y: self.iteration_adjoint(y, psi0, x0)
        self.b = so.fixed_point(f, self.b, xtol=1e-12)
        U = cayley_klein(self.half_time*self.b)
        psi0 = apply_2by2(U, psi0); x0 = hopf(psi0)

        return psi0, x0


    def do_one_step_broyden(self, t, psi0, x0):

        # TODO record residuals for later inspection

        callback = None
        if self.diagnostics:
            callback = self.diagnostics_logger

        #print >> sys.stderr, "direct"
        f = lambda y: self.residue_direct(y, psi0, x0)
        self.b = so.newton_krylov(f, self.b, f_tol=1e-14, callback=callback)
        #res = f(self.b); print np.max(np.max(np.abs(res)))
        U = cayley_klein(self.half_time*self.b)
        psi0 = apply_2by2(U, psi0); x0 = hopf(psi0)

        self.diagnostics_logger.store()
        #print "Iterations: %d." % c.niter
        #c.reset()

        #print >> sys.stderr, "adjoint"
        f = lambda y: self.residue_adjoint(y, psi0, x0)
        self.b = so.newton_krylov(f, self.b, f_tol=1e-14, callback=callback)
        #res = f(self.b); print np.max(np.max(np.abs(res)))
        U = cayley_klein(self.half_time*self.b)
        psi0 = apply_2by2(U, psi0); x0 = hopf(psi0)

        #print "Iterations: %d." % c.niter
        #c.reset()
        self.diagnostics_logger.store()

        return psi0, x0

    do_one_step = do_one_step_broyden



