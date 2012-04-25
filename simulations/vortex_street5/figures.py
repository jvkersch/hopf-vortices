import matplotlib
matplotlib.use('PDF')
matplotlib.rcParams['savefig.dpi'] = 75       # figure dots per inch
matplotlib.rcParams['figure.figsize'] = 6, 4.5  # figure size in inches
matplotlib.rcParams['font.size'] = 15  # size of default fonts
matplotlib.rcParams['figure.subplot.left'] = .15
#matplotlib.rcParams['figure.subplot.right'] = .25
matplotlib.rcParams['figure.subplot.bottom'] = .15
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, ScalarFormatter

from matlab_io import load_variables



def energy_error_figure(times, delta_E_sphere):

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    
    ax.plot(times, delta_E_sphere, label='Variational')#, c='k')
    #ax.plot(times, delta_E_rk4, label='RK2')#, c='k', linestyle='--')
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


def make_figures():


    print "Making figures..."
    times, vortices, energies, moments = \
        load_variables('svs5_poles_sim.mat',
            ['times', 'vortices', 'energies', 'moments'])

    energy_error = energies - energies[0]

    print "\tEnergy behavior."
    fig, _ = energy_error_figure(times, energy_error)
    fig.savefig('svs5-energy-plot.pdf')



if __name__ == '__main__':
    make_figures()
