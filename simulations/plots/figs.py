"""
Matplotlib customizations for PDF figures designed for printing/publishing.

"""

import pylab
import matplotlib


# Customizations to generate nice PDF figures

def set_figure_defaults(size='medium'):
    """
    Sensible defaults for medium-size PDF figures, with extra 
    space on the left/bottom for x/y labels.

    INPUT
    -----

    - `size` (default: `medium`) -- set of default settings to load (one of
    `small`, `medium` or `large`).  Note: `large` currently not supported.

    """

    if size == 'small':
        matplotlib.rcParams['savefig.dpi'] = 75       # figure dots per inch
        matplotlib.rcParams['figure.figsize'] = 4, 3  # figure size in inches
        matplotlib.rcParams['font.size'] = 12  # size of default fonts
        matplotlib.rcParams['figure.subplot.left'] = .20
        matplotlib.rcParams['figure.subplot.bottom'] = .15
    elif size == 'medium':
        matplotlib.rcParams['savefig.dpi'] = 75       # figure dots per inch
        matplotlib.rcParams['figure.figsize'] = 6, 4.5  # figure size in inches
        matplotlib.rcParams['font.size'] = 15  # size of default fonts
        matplotlib.rcParams['figure.subplot.left'] = .15
        matplotlib.rcParams['figure.subplot.bottom'] = .15
    else:
        raise ValueError("No default settings for size '%s'." % size)

def small_legend(loc, size=15):
    """
    Silly helper function to change the size of legend to something more in 
    line with our figure defaults.

    """
    pylab.legend(loc=loc, prop={'size':size})
