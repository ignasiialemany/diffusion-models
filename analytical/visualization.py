"""Animate a time-varying diffusion solution."""

import numpy as np
import matplotlib.pyplot as plt


def animate_1D(G, t, x, bx, axis=None, xlims=None, ylimsf=[0, 1.1], pause=0.1,
        xlabel='$x$ ($\\mu$m)', ylabel='$G(x,t)$'):
    """Animate a 1D time-varying solution.
    G : (N, M) solution
    t : (N) time
    x : (M) space
    bx : barrier locations
    axis : pyplot axis
    xlims : x-axis limits, optional (endpoints of x)
    ylimsf : y-axis limit factors, optional (0 and 1.1 times max(G))
    pause : how long pyplot waits between time steps
    xlabel, ylabel: axis labels
    """

    # prepare figure
    if not axis:
        axis = plt.gca()
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid()
    if xlims:
        xmin, xmax = xlims
    else:
        xmin, xmax = x[0], x[-1]
    axis.set_xlim(xmin=xmin, xmax=xmax)

    # plot initial data
    y = np.full(G[0].shape, np.nan)  # nan values, so nothing shows
    sol = axis.fill_between(x, y, 0, label='Semi-analytical', fc='r', ec='r', alpha=0.5)

    # show barriers
    if len(bx):
        xlines = (bx, bx)
        ylines = (0, 1)
        lines = axis.plot(xlines, ylines, label='barrier', color='k', ls='-')
    else:
        lines = None

    # add legend
    if lines:
        axis.legend((sol, lines[0]), ('Solution', 'Interfaces'))
    else:
        axis.legend((sol), ('Solution'))

    # time loop
    for n, (tn, Gn) in enumerate(zip(t, G)):

        # update plot
        xy = list(map(list, zip(x, Gn)))
        xy.append([x[-1], 0])
        xy.append([x[0], 0])
        verts = [xy]  # only one list
        sol.set_verts(verts, False)  # data
        ymin, ymax = np.array(ylimsf)*np.max(Gn)
        axis.set_ylim(ymin=ymin, ymax=ymax)  # dynamic limit
        axis.set_title('$t_{{{1:d}}} = {0:g}$ms'.format(tn, n))  # :3.f

        # show
        plt.pause(pause)
