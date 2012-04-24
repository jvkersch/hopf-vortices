import sys
import numpy as np
import scipy.optimize as so
from math import floor

np.set_printoptions(precision=10, suppress=True)

from vectors import row_product
from su2_geometry import cayley_klein
from continuous_vortex_system import scaled_gradient_hamiltonian
from array_solver import FSolveArray


class VortexIntegrator:

    def __init__(self, gamma, sigma=0.0, h=1e-1, verbose=False):

        self.gamma = np.array(gamma)
        self.sigma = sigma
        self.h = h
        self.verbose = verbose
        self.N = self.gamma.size

        # Set up nonlinear solvers
        size = (self.N, 3)
        self.solver_direct  = FSolveArray(self.residue_direct, size=size)
        self.solver_adjoint = FSolveArray(self.residue_adjoint, size=size)


    def iteration_direct(self, b, x0):
        """Return update for `b` via direct fixed-point equation."""

        a  = self.h*b
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
    
        a  = self.h*b
        x1 = cayley_klein(a, x0)
    
        gradH = scaled_gradient_hamiltonian(self.gamma, (x0+x1)/2., self.sigma)
        dot   = np.sum(x1*gradH, axis=1)

        return 1./4*np.cross(np.cross(gradH, x1, axis=1) + row_product(dot, a), 
                             x1, axis=1)


    def residue_adjoint(self, b, x0):
        """Residue for adjoint iteration."""
        return b - self.iteration_adjoint(b, x0)


    def integrate(self, x0, tmax=50., numpoints=100):

        num_inner = int(floor(tmax/(2*self.h*numpoints)))

        t = 0

        # Output variables
        vortices = np.zeros((numpoints, ) + x0.shape)
        times = np.zeros(numpoints)

        if self.verbose:
            print >> sys.stderr, "Entering integration loop"            

        b = np.zeros((self.N, 3), dtype=np.double)
        for k in xrange(0, numpoints):
            print >> sys.stderr, '.',
            for _ in xrange(0, num_inner):

                # Step with direct method             
                b = self.solver_direct.fsolve(b, args=x0)
                x0 = cayley_klein(self.h*b, x0)

                # Step with adjoint method
                b = self.solver_adjoint.fsolve(b, args=x0)
                x0 = cayley_klein(self.h*b, x0)

                # Update time step
                t += 2*self.h

            # Save output
            vortices[k, :, :] = x0
            times[k] = t

        print >> sys.stderr, '\n'
        return vortices, times


if __name__ == '__main__':
   from matlab_io import load_ic_from_matfile 
   gamma, x0, sigma = load_ic_from_matfile('collapse3py.mat')

   #print gamma
   #print x0
   #print sigma

   setup = "from . import VortexIntegrator; v = VortexIntegrator(gamma, sigma=0.0, h=1e-1, verbose=False)"
   stmt  = "v.integrate(x0)"

   import timeit
   t = timeit.Timer(stmt=stmt, setup=setup)
   print t.timeit(number=3)
