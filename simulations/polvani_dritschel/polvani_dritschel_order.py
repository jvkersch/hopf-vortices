from hopf.simulation import Simulation
from numpy import logspace


## Simulations for different time scales (sphere integrator)

s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')

for n, h in enumerate(logspace(-3, -1, 10)):
    s.run_simulation(tmax=50, h=h, sim='sphere')
    s.post_process()

    filename = 'data/pd_sphere_order_%d.mat' % n
    s.save_results(filename)
