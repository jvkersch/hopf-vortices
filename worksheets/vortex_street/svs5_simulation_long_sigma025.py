from hopf.simulation import Simulation

# Sphere integrator
s = Simulation()
s.load_initial_conditions('svs5_poles.mat')
s.sigma = 0.25
s.run_simulation(h=0.25, tmax=10000, numpoints=1000, sim='sphere')
s.post_process()
s.save_results('data/svs5_poles_sphere_long.mat')

# RK4 integrator
s = Simulation()
s.load_initial_conditions('svs5_poles.mat')
s.sigma = 0.25
s.run_simulation(h=0.25, tmax=10000, numpoints=1000, sim='rk4')
s.post_process()
s.save_results('data/svs5_poles_rk4_long.mat')

# LP integrator
s = Simulation()
s.load_initial_conditions('svs5_poles.mat')
s.sigma = 0.25
s.run_simulation(h=0.25, tmax=10000, numpoints=1000, sim='lie-poisson')
s.post_process()
s.save_results('data/svs5_poles_lp_long.mat')


# MP integrator
s = Simulation()
s.load_initial_conditions('svs5_poles.mat')
s.sigma = 0.25
s.run_simulation(h=0.25, tmax=10000, numpoints=1000, sim='midpoint')
s.post_process()
s.save_results('data/svs5_poles_mp_long.mat')
