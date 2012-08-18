from hopf.simulation import Simulation
from numpy import logspace
import scipy

## Simulations for different time scales (sphere integrator)

s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')

for n, h in enumerate(logspace(-3, -1, 10)):
    s.run_simulation(tmax=10, h=h, sim='sphere', diagnostics=True)
    s.post_process()

    filename = 'data/pd_sphere_order_mu_%d.mat' % n
    s.save_results(filename)


## Simulations for different time scales (sphere-mu integrator)

s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')

for n, h in enumerate(logspace(-3, -1, 10)):
    s.run_simulation(tmax=10, h=h, sim='sphere', diagnostics=True)
    s.post_process()

    filename = 'data/pd_sphere_mu_order_mu_%d.mat' % n
    s.save_results(filename)

