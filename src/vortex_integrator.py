import numpy as np
from su2_geometry import cayley_klein

class VortexSystem:
    h = 1e-1

def optimization_direct(B, X0, vortex_system):
    
    A = vortex_system.h*B
    U = cayley_klein(A)
    X1 = np.einsum('abc, bdc -> adc', U, X0)

def optimization_adjoint():
    pass 


if __name__ == '__main__':
    A = np.array([[2, 1, 3], [4, 6, 6]], dtype=np.double)
    v = VortexSystem()
    
    print optimization_direct(A, 2*A, v)
