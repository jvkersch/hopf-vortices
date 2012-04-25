import matplotlib
matplotlib.use('PDF')
#matplotlib.rcParams['savefig.dpi'] = 75       # figure dots per inch
#matplotlib.rcParams['figure.figsize'] = 6, 4.5  # figure size in inches
matplotlib.rcParams['font.size'] = 15  # size of default fonts
#matplotlib.rcParams['figure.subplot.left'] = .25
#matplotlib.rcParams['figure.subplot.right'] = .25
matplotlib.rcParams['figure.subplot.bottom'] = .15
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, ScalarFormatter

from matlab_io import load_variables



def energy_error_figure(times, delta_E_sphere, delta_E_rk4):

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    
    ax.plot(times, delta_E_sphere, label='Variational')#, c='k')
    ax.plot(times, delta_E_rk4, label='RK2')#, c='k', linestyle='--')
    ax.autoscale(axis='x', tight=True)

    sfmt = ScalarFormatter()
    sfmt.set_scientific(True)
    sfmt.set_powerlimits((-2, 4))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(sfmt)

    ax.set_xlabel(r'$t$', fontsize=20)
    ax.set_ylabel(r'$\Delta E$', fontsize=20)

    ax.legend(loc='lower left')

    #fig.subplots_adjust(bottom=.15)

    return fig, ax


def momentum_error_figure(times, delta_M_sphere, delta_M_rk4):

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    
    ax.plot(times, delta_M_sphere, label='Variational', c='k')
    ax.plot(times, delta_M_rk4, label='RK2', c='k', linestyle='--')
    ax.autoscale(axis='x', tight=True)

    sfmt = ScalarFormatter()
    sfmt.set_scientific(True)
    sfmt.set_powerlimits((-2, 4))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(sfmt)

    ax.set_xlabel(r'$t$', fontsize=20)
    ax.set_ylabel(r'$\Delta M$', fontsize=20)

    ax.legend(loc='upper left')

    #fig.subplots_adjust(bottom=.15)

    return fig, ax

def make_figures():
    from process import load_ic_from_matfile, load_variables_from_matfile
    from figures import (vortex_figure, energy_error_figure, 
                         perimeter_figure, moment_error_figure)

    print "Making figures..."
    ic_file = 'data/svs5poles.mat'
    times, vortices, energy_err, moment_err = \
        load_variables_from_matfile(ic_file,
            ['times', 'vortices', 'energy_err', 'moment_err'])

    print "\tFirst vortex components"
    x = vortices[:, 0, 0]
    y = vortices[:, 1, 0]
    z = vortices[:, 2, 0]
    fig, _ = vortex_figure(times, x, y, z)
    fig.savefig('data/svs5-vortex1.pdf')

    
    print "\tEnergy behavior."
    fig, _ = energy_error_figure(times, energy_err)
    fig.savefig('data/svs5-energy.pdf')


    print "\tMoment error."
    fig, ax = moment_error_figure(times, moment_err)
    fig.savefig('data/svs5-moment.pdf')





if __name__ == '__main__':
    
    [times, e_sphere, m_sphere] = load_variables_from_matfile(
        'data/street_coarse_sphere.mat', 
        ['times', 'energy_err', 'moment_err'])

    [times, e_rk2, m_rk2] = load_variables_from_matfile(
        'data/street_coarse_rk2.mat', 
        ['times', 'energy_err', 'moment_err'])

    from numpy.linalg import norm

    mom_sphere = [norm(mom) for mom in m_sphere.T]
    mom_rk2    = [norm(mom) for mom in m_rk2.T]

    print len(mom_sphere), len(mom_rk2)
    print len(times)

    fig, ax = momentum_error_figure(times, mom_sphere, mom_rk2)
    fig.savefig('data/vortex_street_moment.pdf')

