import numpy as np
from vectors import row_product
from su2_geometry import cayley_klein
from continuous_vortex_system import scaled_gradient_hamiltonian

from math import floor
import scipy.optimize as so

import sys

np.set_printoptions(precision=10, suppress=True)


class VortexIntegrator:

    def __init__(self, gamma, x0, sigma=0.0, h=1e-1, verbose=False):

        self.gamma = np.array(gamma)
        self.x0 = np.array(x0) 
        self.sigma = sigma
        self.h = h
        self.verbose = verbose

        self.N = self.gamma.size
        n, ndim = self.x0.shape
        if n != self.N:
            raise ValueError, "Number of vortex strengths and vortices" \
                " does not agree."
        if ndim != 3:
            raise ValueError, "Vortex locations not specified as 3-vectors."


    def iteration_direct(self, b, x0):
        """Return update for `b` via direct fixed-point equation."""

        a = self.h*b
        x1 = cayley_klein(a, x0)

        gradH = scaled_gradient_hamiltonian(self.gamma, (x0+x1)/2., self.sigma)
        dot   = np.sum(x0*gradH, axis=1)

        return 1./4*np.cross(np.cross(gradH, x0, axis=1) - row_product(dot, a), 
                             x0, axis=1)


    def iteration_adjoint(self, b, x0):
        """Return update for `b` via adjoint fixed-point equation."""
    
        a = self.h*b
        x1 = cayley_klein(a, x0)
    
        gradH = scaled_gradient_hamiltonian(self.gamma, (x0+x1)/2., self.sigma)
        dot   = np.sum(x1*gradH, axis=1)

        return 1./4*np.cross(np.cross(gradH, x1, axis=1) + row_product(dot, a), 
                             x1, axis=1)


    def flattened_optim_direct(self, b, x0):
        b.shape = (self.N, 3)
        res = b - self.iteration_direct(b, x0)
        res.shape = 3*self.N
        return res


    def flattened_optim_adjoint(self, b, x0):
        b.shape = (self.N, 3)
        res = b - self.iteration_adjoint(b, x0)
        res.shape = 3*self.N
        return res


    def integrate(self, tmax=50., numpoints=100):

        num_inner = int(floor(tmax/(self.h*numpoints)))

        t = 0

        self.vortices = np.zeros((numpoints, ) + self.x0.shape)
        self.times = np.zeros(numpoints)

        if self.verbose:
            print >> sys.stderr, "Entering integration loop"            

        b = np.zeros(3*self.N, dtype=np.double)
        for k in xrange(0, numpoints):
            print >> sys.stderr, '.',
            for _ in xrange(0, num_inner):
             
                # Step with direct method
                b = so.fsolve(self.flattened_optim_direct, b, 
                              args=self.x0, xtol=1e-12)
                b.shape = (self.N, 3)
                self.x0 = cayley_klein(self.h*b, self.x0)
                b.shape = 3*self.N

                # Step with adjoint method
                b = so.fsolve(self.flattened_optim_adjoint, b, args=self.x0)
                b.shape = (self.N, 3)                
                self.x0 = cayley_klein(self.h*b, self.x0)
                b.shape = 3*self.N

  
                t += 2*self.h

            # Save output
            self.vortices[k, :, :] = self.x0
            self.times[k] = t

        print >> sys.stderr, '\n'


if __name__ == '__main__':
   from matlab_io import load_ic_from_matfile 
   gamma, x0, sigma = load_ic_from_matfile('collapse3py.mat')

   print gamma
   print x0
   print sigma

   v = VortexIntegrator(gamma, x0, sigma=0.0, h=1e-1, verbose=False)
   v.integrate()
