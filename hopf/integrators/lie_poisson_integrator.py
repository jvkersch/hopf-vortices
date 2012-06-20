import numpy as np
import sys
from scipy import linalg
from scipy.optimize import fsolve, fixed_point, broyden1

from ..lie_algebras.adjoint import Ad
from ..vortices.continuous_vortex_system import scaled_gradient_hamiltonian
from .vectorized_so3 import vector_hatmap, vector_invhat
from .generic_integrator import GenericIntegrator

from .diagnostics import BroydenDiagnostics

# Legacy class from way back in the days.  Has the same calling conventions as
# the other integrators, but that's about it.


class LiePoissonIntegrator(GenericIntegrator):
    def __init__(self, gamma, sigma, h, verbose=False, diagnostics=False):

        self.gamma = np.array(gamma)
        self.sigma = sigma
        self.N = len(self.gamma)

        # Keep track of nonlinear convergence
        self.diagnostics = diagnostics
        self.diagnostics_logger = BroydenDiagnostics()

        GenericIntegrator.__init__(self, h, verbose)

    def do_one_step(self, t, X0):

        rho0 = vector_hatmap(X0, self.gamma)
        grad = self.gradient_hamiltonian(rho0)

        callback = None
        if self.diagnostics:
            callback = self.diagnostics_logger

        def optimization_function(rho1):
            s = grad + self.gradient_hamiltonian(rho1)
            res  = np.empty((self.N, 3, 3))
            for n in range(0, self.N):               
                g = linalg.expm(-self.h/2*s[n, :, :])
                res[n, :, :] = rho1[n, :, :] - Ad(g, rho0[n, :, :])
            return res


        #d = Diagnostics()

        rho1 = broyden1(optimization_function, rho0, f_tol=1e-14, callback=callback)
        self.diagnostics_logger.store()


        #print d.n_current


        return vector_invhat(rho1, self.gamma)

    def gradient_hamiltonian(self, rho):
        
        x = vector_invhat(rho, self.gamma)
        rhs = scaled_gradient_hamiltonian(self.gamma, x, self.sigma)
        return vector_hatmap(rhs, np.ones(self.N))
