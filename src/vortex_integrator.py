import numpy as np
from vectors import row_product
from su2_geometry import cayley_klein
from continuous_vortex_system import scaled_gradient_hamiltonian

class VortexSystem:
    """Prototype struct to hold vortex system parameters."""

    h = 1e-1
    N = 2
    gamma = np.ones(2) 
    sigma = 0.



def iteration_direct(b, x0, vortex_system):
    """Return update for `b` via direct fixed-point equation."""

    a = vortex_system.h*b
    x1 = cayley_klein(a, x0)
    
    gradH = scaled_gradient_hamiltonian(
        vortex_system.gamma, 
        (x0+x1)/2., 
        vortex_system.sigma)

    dot = np.sum(x0*gradH, axis=1)

    return 1./4*np.cross(np.cross(gradH, x0, axis=1) - row_product(dot, a), 
                         x0, axis=1)
                         

def iteration_adjoint(b, x0, vortex_system):
    """Return update for `b` via adjoint fixed-point equation."""
    
    a = vortex_system.h*b
    x1 = cayley_klein(a, x0)
    
    gradH = scaled_gradient_hamiltonian(
        vortex_system.gamma, 
        (x0+x1)/2., 
        vortex_system.sigma)

    dot = np.sum(x1*gradH, axis=1)

    return 1./4*np.cross(np.cross(gradH, x1, axis=1) + row_product(dot, a), 
                         x1, axis=1)


    
