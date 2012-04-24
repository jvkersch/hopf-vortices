import sys
sys.path.append('../src')

import numpy as np
from vortex_integrator import VortexIntegrator
from matlab_io import load_initial_conditions, save_variables_to_matfile
from continuous_vortex_system import vortex_hamiltonian, vortex_moment


def make_output_filename(base_name):
    import os.path
    head, tail = os.path.split(base_name)
    root, ext  = os.path.splitext(tail)
    new_filename = root + '_sim' + ext
    return os.path.join(head, new_filename)

class Simulation:

    def load_initial_conditions(self, filename):

        self.ic_file = filename

        try: 
            self.gamma, self.X0, self.sigma = \
                load_initial_conditions(filename)
        except OSError:
            print "Could not load initial conditions from %s." \
                % filename
            raise

        self.N = self.X0.shape[0]


    def run_simulation(self, tmax=20., h=.1, d=.0, numpoints=100, sim='sphere'):
        if self.gamma is None or self.X0 is None:
            raise ValueError, "Initial conditions not set."

        v = VortexIntegrator(self.gamma, self.sigma, h)
        [self.vortices, self.times] = v.integrate(self.X0, tmax, numpoints)


    def post_process(self):

        n = len(self.times)
        self.energies = np.zeros(n)
        self.moments = np.zeros((n, 3))

        for k in xrange(0, n):
            vortex = self.vortices[k, :, :]
            self.energies[k] = \
                vortex_hamiltonian(self.gamma, vortex, self.sigma)
            self.moments[k] = vortex_moment(self.gamma, vortex)


    def save_results(self, outfile=None):
        
        if self.times is None or self.vortices is None:
            raise ValueError, "Simulation must be run before "\
                " saving results."

        output_filename = make_output_filename(self.ic_file)
        save_variables_to_matfile(output_filename,
                                  {'vortices': self.vortices, 
                                   'times': self.times,
                                   'energies': self.energies,
                                   'moments': self.moments})


if __name__ == '__main__':
    
                                       
    s = Simulation()
    s.load_initial_conditions(sys.argv[1])
    s.run_simulation()
    s.post_process()
    s.save_results()
