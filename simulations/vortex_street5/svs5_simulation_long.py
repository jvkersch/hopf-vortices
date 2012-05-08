from simulation import Simulation

# Sphere integrator
s = Simulation()
s.load_initial_conditions('svs5_poles.mat')
s.sigma = 0.25
s.run_simulation(h=0.3, tmax=1000, sim='sphere')
s.post_process()
s.save_results('svs5_poles_sphere_long_025.mat')

# RK4 integrator
s = Simulation()
s.load_initial_conditions('svs5_poles.mat')
s.sigma = 0.25
s.run_simulation(h=0.3, tmax=1000, sim='rk4')
s.post_process()
s.save_results('svs5_poles_rk4_long_025.mat')


# Sphere integrator
s = Simulation()
s.load_initial_conditions('svs5_poles.mat')
s.sigma = 0.5
s.run_simulation(h=0.3, tmax=1000, sim='sphere')
s.post_process()
s.save_results('svs5_poles_sphere_long_050.mat')

# RK4 integrator
s = Simulation()
s.load_initial_conditions('svs5_poles.mat')
s.sigma = 0.5
s.run_simulation(h=0.3, tmax=1000, sim='rk4')
s.post_process()
s.save_results('svs5_poles_rk4_long_050.mat')