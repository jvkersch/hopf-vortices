import sys
import numpy as np
import scipy.optimize as so
#from math import ceil

from generic_integrator import GenericIntegrator
from ..vortices.continuous_vortex_system import vortex_rhs

# TODO Method docstrings are horrible
# TODO Can we bring some clarity into the morass of different optimization methods?

from ..util.vectors import row_product
from ..util.array_solver import FSolveArray
from ..lie_algebras.su2_geometry import (cayley_klein, apply_2by2, hopf, 
                                         inverse_hopf)
from ..vortices.continuous_vortex_system import scaled_gradient_hamiltonian



class VortexIntegrator:

    def __init__(self, gamma, sigma=0.0, h=1e-1, verbose=False):

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

        # TODO record residuals for later inspection

        f = lambda y: self.residue_direct(y, psi0, x0)
        self.b = so.broyden1(f, self.b, f_tol=1e-15)
        #res = f(self.b); print np.max(np.max(np.abs(res)))
        U = cayley_klein(self.half_time*self.b)
        psi0 = apply_2by2(U, psi0); x0 = hopf(psi0)

        f = lambda y: self.residue_adjoint(y, psi0, x0)
        self.b = so.broyden1(f, self.b, f_tol=1e-15)
        #res = f(self.b); print np.max(np.max(np.abs(res)))
        U = cayley_klein(self.half_time*self.b)
        psi0 = apply_2by2(U, psi0); x0 = hopf(psi0)

        return psi0, x0


    def do_one_step_old(self, x0):

        # Step with direct method             
        self.b = self.solver_direct.fsolve(self.b, args=x0)
        x0 = cayley_klein(self.half_time*self.b, x0)

        res = self.residue_direct(self.b, x0)
        self.res_direct = np.max(np.max(np.abs(res)))

        # Step with adjoint method
        self.b = self.solver_adjoint.fsolve(self.b, args=x0)
        x0 = cayley_klein(self.half_time*self.b, x0)

        res = self.residue_adjoint(self.b, x0)
        self.res_adjoint = np.max(np.max(np.abs(res)))

        return x0


