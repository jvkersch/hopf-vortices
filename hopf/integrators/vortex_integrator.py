#import sys
import numpy as np
import scipy.optimize as so
#from math import ceil

from generic_integrator import GenericIntegrator
from ..vortices.continuous_vortex_system import vortex_rhs

# TODO Method docstrings are horrible
# TODO Can we bring some clarity into the morass of different optimization methods?

from ..util.vectors import row_product
from ..util.array_solver import FSolveArray
from ..lie_algebras.su2_geometry import cayley_klein
from ..vortices.continuous_vortex_system import scaled_gradient_hamiltonian



class VortexIntegrator(GenericIntegrator):

    def __init__(self, gamma, sigma=0.0, h=1e-1, verbose=False):

        self.gamma = np.array(gamma)
        self.sigma = sigma
        self.N = self.gamma.size

        # Set up nonlinear solvers
        size = (self.N, 3)
        self.solver_direct  = FSolveArray(self.residue_direct, size=size)
        self.solver_adjoint = FSolveArray(self.residue_adjoint, size=size)

        # Initial choice for update element
        self.b = np.zeros((self.N, 3), dtype=np.double)

        # Initialize base class
        # Note that since this is a composition method, each iteration 
        # will take two steps of size h, resulting in an overall time 
        # step per iteration of 2*h
        self.half_time = h
        GenericIntegrator.__init__(self, 2*h, verbose)


    def iteration_direct(self, b, x0):
        """Return update for `b` via direct fixed-point equation."""

        a  = self.half_time*b
        x1 = cayley_klein(a, x0)

        gradH = scaled_gradient_hamiltonian(self.gamma, (x0+x1)/2., self.sigma)
        dot   = np.sum(x0*gradH, axis=1)

        return 1./4*np.cross(np.cross(gradH, x0, axis=1) - row_product(dot, a), 
                             x0, axis=1)


    def residue_direct(self, b, x0):
        """Residue for direct iteration."""
        return b - self.iteration_direct(b, x0)


    def iteration_adjoint(self, b, x0):
        """Return update for `b` via adjoint fixed-point equation."""
    
        a  = self.half_time*b
        x1 = cayley_klein(a, x0)
    
        gradH = scaled_gradient_hamiltonian(self.gamma, (x0+x1)/2., self.sigma)
        dot   = np.sum(x1*gradH, axis=1)

        return 1./4*np.cross(np.cross(gradH, x1, axis=1) + row_product(dot, a), 
                             x1, axis=1)


    def residue_adjoint(self, b, x0):
        """Residue for adjoint iteration."""
        return b - self.iteration_adjoint(b, x0)


    def do_one_step(self, x0):

        # Step with direct method             
        self.b = self.solver_direct.fsolve(self.b, args=x0)
        x0 = cayley_klein(self.half_time*self.b, x0)

        # Step with adjoint method
        self.b = self.solver_adjoint.fsolve(self.b, args=x0)
        x0 = cayley_klein(self.half_time*self.b, x0)

        return x0


