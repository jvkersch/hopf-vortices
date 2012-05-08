setup = """
import sys
sys.path.append('..')

from continuous_vortex_system import scaled_gradient_hamiltonian
from matlab_io import load_variables_from_matfile


[x0, gamma] = load_variables_from_matfile('big_gradient_system.mat', 
                                          ['x0', 'gamma'])

gamma = gamma[0]
sigma = 0
"""

if __name__ == '__main__':
    from timeit import Timer
    t = Timer("scaled_gradient_hamiltonian(gamma, x0, sigma)", setup)
    print t.timeit(number=10)
