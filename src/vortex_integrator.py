import numpy as np
from vectors import row_product
from su2_geometry import cayley_klein
from continuous_vortex_system import scaled_gradient_hamiltonian

class VortexSystem:
    h = 1e-1
    N = 2
    gamma = np.ones(2) 
    sigma = 0.

def optimization_direct(b, x0, vortex_system):
    
    a = vortex_system.h*b
    x1 = cayley_klein(a, x0)
    
    gradH = scaled_gradient_hamiltonian(
        vortex_system.gamma, 
        (x0+x1)/2., 
        vortex_system.sigma)

    dot = np.sum(x0*gradH, axis=1)

    return 1./4*np.cross(np.cross(gradH, x0, axis=1) + row_product(dot, a), 
                         x0, axis=1)
                         
def res(b, x0, vortex_system):
    a = vortex_system.h*b
    r = optimization_direct(a, x0, vortex_system)

    return 2*row_product(vortex_system.gamma, b - r)


def optimization_adjoint():
    pass


if __name__ == '__main__':
    b = np.array([[2, 1, 3], [4, 6, 6]], dtype=np.double)
    v = VortexSystem()
    
    print res(b, 2*b, v)
