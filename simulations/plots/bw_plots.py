"""
Override most common matplotlib commands to produce b/w plots.

"""

from pylab import plot, semilogx, semilogy, loglog


def _patch_plot_method_bw(method):
    """
    Returns a patched version of a given plotting method which uses b/w 
    line styles.

    """
    def patched_method(*args, **kwargs):
        kwargs['color'] = 'black'
        method(*args, **kwargs)

    return patched_method

# Provide b/w version of plot methods
plot = _patch_plot_method_bw(plot)
loglog = _patch_plot_method_bw(loglog)
semilogx = _patch_plot_method_bw(semilogx)
semilogy = _patch_plot_method_bw(semilogy)

