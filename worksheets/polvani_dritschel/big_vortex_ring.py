import numpy as np
from hopf.simulation import Simulation
#from hopf.lie_algebras.su2_geometry import inverse_hopf
from hopf.util.matlab_io import save_initial_conditions

# Equatorial vortex ring, 40 point vortices of strength 1/8
N = 40
gamma = 1./8*np.ones(40)
sigma = 0.1

z = .9 # Height above equatorial plane
r = (1 - z**2)**.5

theta = np.linspace(0, 2*np.pi, N, endpoint=False)

X = np.array([r*np.cos(theta), r*np.sin(theta), z*np.ones(N)]).T

print X



#phi = inverse_hopf(X)

# Save initial conditions
save_initial_conditions('equatorial_ring_40.mat', gamma, X, sigma)

# Sphere integrator
s = Simulation()
s.load_initial_conditions('equatorial_ring_40.mat')
s.run_simulation(tmax=50, sim='sphere-midpoint')
s.post_process()
s.save_results('data/equatorial_sphere.mat')

# RK4 integrator
s = Simulation()
s.load_initial_conditions('equatorial_ring_40.mat')
s.run_simulation(tmax=50, sim='rk4')
s.post_process()
s.save_results('data/equatorial_rk4.mat')

# Midpoint integrator
s = Simulation()
s.load_initial_conditions('equatorial_ring_40.mat')
s.run_simulation(tmax=50, sim='midpoint')
s.post_process()
s.save_results('data/equatorial_mp.mat')

# Lie-Poisson integrator
s = Simulation()
s.load_initial_conditions('equatorial_ring_40.mat')
s.run_simulation(tmax=50, sim='lie-poisson')
s.post_process()
s.save_results('data/equatorial_lp.mat')

