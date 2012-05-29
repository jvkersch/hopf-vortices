from hopf.simulation import Simulation

# Sphere integrator
s = Simulation()
s.load_initial_conditions('svs5_poles.mat')
s.run_simulation(tmax=100, sim='sphere')
s.post_process()
s.save_results('data/svs5_poles_sphere.mat')

# RK4 integrator
s = Simulation()
s.load_initial_conditions('svs5_poles.mat')
s.run_simulation(tmax=100, sim='rk4')
s.post_process()
s.save_results('data/svs5_poles_rk4.mat')
