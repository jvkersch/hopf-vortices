from hopf.simulation import Simulation

##### Long time

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

# Midpoint integrator
s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')
s.run_simulation(tmax=1000, sim='midpoint')
s.post_process()
s.save_results('data/pd_mp_long.mat')

# Lie-Poisson integrator
s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')
s.run_simulation(tmax=1000, sim='lie-poisson')
s.post_process()
s.save_results('data/pd_lp_long.mat')


##### Short time

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

# Midpoint integrator
s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')
s.run_simulation(tmax=100, sim='midpoint')
s.post_process()
s.save_results('data/pd_mp_short.mat')

# Lie-Poisson integrator
s = Simulation()
s.load_initial_conditions('pd_initial_conditions.mat')
s.run_simulation(tmax=100, sim='lie-poisson')
s.post_process()
s.save_results('data/pd_lp_short.mat')
