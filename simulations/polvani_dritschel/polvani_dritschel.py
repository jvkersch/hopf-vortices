from hopf.simulation import Simulation

## Long time

# Sphere integrator
s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')
s.run_simulation(tmax=1000, sim='sphere')
s.post_process()
s.save_results('data/pd_sphere_long.mat')

# RK4 integrator
s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')
s.run_simulation(tmax=1000, sim='rk4')
s.post_process()
s.save_results('data/pd_rk4_long.mat')

## Short time

# Sphere integrator
s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')
s.run_simulation(tmax=100, sim='sphere')
s.post_process()
s.save_results('data/pd_sphere_short.mat')

# RK4 integrator
s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')
s.run_simulation(tmax=100, sim='rk4')
s.post_process()
s.save_results('data/pd_rk4_short.mat')
