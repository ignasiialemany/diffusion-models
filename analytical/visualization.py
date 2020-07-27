"""Animate a time-varying diffusion solution."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animate


def animate_1D(G, t, x, bx, axis=None, fig=None, xlims=None, ylimsf=[0, 1.1],
        xlabel='$x$ ($\\mu$m)', ylabel='$G(x,t)$'):
    """Animate a 1D time-varying solution.
    G : (N, M) solution
    t : (N) time
    x : (M) space
    bx : barrier locations
    axis : pyplot axis
    fig : pyplot figure
    xlims : x-axis limits, optional (endpoints of x)
    ylimsf : y-axis limit factors, optional (0 and 1.1 times max(G))
    xlabel, ylabel: axis labels
    """

    # prepare figure
    if axis is None:
        axis = plt.gca()
    if fig is None:
        fig = plt.gcf()
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid()
    if xlims is None:
        xlims = (x[0], x[-1])
    axis.set_xlim(xlims)

    # plot initial data
    fill = axis.fill_between([], [], # empty data
                             label='Semi-analytical', fc='r', ec='r', alpha=0.5)

    # show barriers
    if len(bx):
        xlines = (bx, bx)
        ylines = (0, 1)
        lines = axis.plot(xlines, ylines, label='barrier', color='k', ls='-')
    else:
        lines = None

    # add legend
    if lines:
        axis.legend((fill, lines[0]), ('Solution', 'Interfaces'))
    else:
        axis.legend((fill), ('Solution'))

    # data
    data = list(zip(t, G))
    def update_fill(n, y0=0):

        tn, yn = data[n]

        ymin, ymax = np.array(ylimsf)*np.max(yn)
        axis.set_ylim(ymin=ymin, ymax=ymax)  # dynamic limit
        axis.set_title('$t_{{{1:d}}} = {0:g}$ms'.format(tn, n))

        xy = list(map(list, zip(x, yn)))
        xy.append([x[-1], y0])
        xy.append([x[0], y0])
        verts = [xy]  # only one list
        fill.set_verts(verts, False)  # data
        return fill

    # make animation
    opts = dict(repeat=False, blit=False)
    animation = mpl_animate.FuncAnimation(fig, update_fill, frames=len(data), **opts)
    return animation
