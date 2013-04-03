import numpy as np
from .integrators.vortex_integrator import VortexIntegrator
from .integrators.rk4_integrator import RK4VortexIntegrator
from .integrators.lie_poisson_integrator import LiePoissonIntegrator
from .integrators.midpoint_integrator import MidpointIntegrator
from .integrators.vortex_integrator_mu import VortexIntegrator_mu
from .integrators.vortex_integrator_twostep import VortexIntegratorTwostep
from .util.matlab_io import (load_initial_conditions, save_variables, 
                             load_variables)
from .vortices.continuous_vortex_system import vortex_hamiltonian, vortex_moment


def make_output_filename(base_name, postfix):
    import os.path
    head, tail = os.path.split(base_name)
    root, ext  = os.path.splitext(tail)
    new_filename = root + '_' + postfix + ext
    return os.path.join(head, new_filename)


class Simulation:

    def load_initial_conditions(self, filename):

        self.ic_file = filename

        try: 
            self.gamma, self.X0, self.sigma = \
                load_variables(filename, ['gamma', 'X0', 'sigma'])
        except OSError:
            print "Could not load initial conditions from %s." \
                % filename
            raise

        self.N = self.X0.shape[0]


    def run_simulation(self, tmax=20., h=.1, numpoints=100, 
                       sim='sphere', diagnostics=False):

        if self.gamma is None or self.X0 is None:
            raise ValueError, "Initial conditions not set."

        # TODO: add logging everywhere

        self.sim_type = sim
        if sim == 'sphere':
            v = VortexIntegrator(self.gamma, self.sigma, h)
        elif sim == 'sphere-momentum':
            v = VortexIntegrator(self.gamma, self.sigma, h, compute_momentum=True)
        elif sim == 'rk4':
            v = RK4VortexIntegrator(self.gamma, self.sigma, h)
        elif sim == 'lie-poisson':
            v = LiePoissonIntegrator(self.gamma, self.sigma, h, diagnostics=diagnostics)
        elif sim == 'midpoint':
            v = MidpointIntegrator(self.gamma, self.sigma, h, diagnostics=diagnostics)
        elif sim == 'sphere-mu':
            v = VortexIntegrator_mu(self.gamma, self.sigma, h, diagnostics=diagnostics)
        elif sim == 'sphere-twostep':
            v = VortexIntegratorTwostep(self.gamma, self.sigma, h)
        elif sim == 'sphere-twostep-momentum':
            v = VortexIntegratorTwostep(self.gamma, self.sigma, h, compute_momentum=True)
        else:
            raise ValueError, "Simulator %s not available." % sim
            
        output = v.integrate(self.X0, tmax, numpoints, full_output=True)

        self.vortices = output[0]
        self.times = output[1]

        if sim == 'sphere-momentum':
            self.momentum = output[2]

        self.diagnostics=diagnostics
        if diagnostics:
            self.number_iterations, self.residues = \
                np.array(v.diagnostics_logger.number_iterations), \
                np.array(v.diagnostics_logger.residues)


    def post_process(self):

        n = len(self.times)
        self.energies = np.zeros(n)
        self.moments = np.zeros((n, 3))

        for k in xrange(0, n):
            vortex = self.vortices[k, :, :]
            self.energies[k] = \
                vortex_hamiltonian(self.gamma, vortex, self.sigma)
            self.moments[k] = vortex_moment(self.gamma, vortex)


    def save_results(self, output_filename=None):
        
        if self.times is None or self.vortices is None:
            raise ValueError, "Simulation must be run before "\
                " saving results."

        if output_filename is None:
            output_filename = make_output_filename(self.ic_file, self.sim_type)

        variables = {'vortices': self.vortices, 'times': self.times,
                     'energies': self.energies, 'moments': self.moments}

        if self.diagnostics:
            variables['number_iterations'] = self.number_iterations
            variables['residues'] = self.residues

        save_variables(output_filename, variables)
                       

if __name__ == '__main__':
    
                                       
    s = Simulation()
    s.load_initial_conditions(sys.argv[1])
    s.run_simulation()
    s.post_process()
    s.save_results()
