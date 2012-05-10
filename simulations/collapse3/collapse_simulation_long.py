from hopf.simulation import Simulation

# Sphere integrator
s = Simulation()
s.load_initial_conditions('collapse3.mat')

# Medium regularization
s.sigma = 0.10
s.run_simulation(tmax=500, numpoints=1000, sim='sphere')
s.post_process()
s.save_results('collapse3_sphere_sigma10_sim_long.mat')

# Runge-Kutta, medium regularization
s.sigma = 0.10
s.run_simulation(tmax=500, numpoints=1000, sim='rk4')
s.post_process()
s.save_results('collapse3_rk4_sigma10_sim_long.mat')
