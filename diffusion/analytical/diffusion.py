"""Routines to calculate 1D solutions to the diffusion problem."""

import math

import numpy as np


def solution_1D(t, x, idx0, *args, **kwargs):
    """Calculate the fundamental 1D solution.

    t := (scalar) time
    x = (array) position
    idx0 = (scalar) index such that x0 = x[idx0]
    soltype & kwargs (only for t>0):
        'free', D0
        'plates', D0, L, N, [sumtype], [clip, thresh]
        'arbitrary', lambdas, nus, [sumtype], [clip, thresh]
    """

    if t == 0:  # initial condition

        # By definition, this is G(x, t0; x0). If the numerical approximation is
        # desired, one should call this function with t = `realmin`, `eps`, etc.
        peak = kwargs.pop('peak', math.inf)
        G = diracdelta(x, idx0, peak=peak)

    elif t > 0:

        soltype = args[0]
        if soltype == 'free':

            D0 = kwargs.pop('D0')
            G = gaussian(x, idx0, D0, t)

        elif soltype == 'plates':

            D0 = kwargs.pop('D0')  # constant diffusivity between plates
            L = kwargs.pop('L')  # spacing between plates
            N = kwargs.pop('N', 100)  # number of terms in the series

            # set the eigenvalues and eigenmodes from the known solution
            n = np.arange(0, N).reshape(N, 1)  # column vector of n-terms
            a_n = np.sqrt(1/L) * np.ones(n.shape)
            a_n[1:] = np.sqrt(2) * a_n[1:]
            nu_n     = np.cos( n*np.pi/L * x ) * a_n
            lambda_n =        (n*np.pi/L)**2 * D0

            G = series(lambda_n, nu_n, idx0, t, **kwargs)

        elif soltype == 'arbitrary':

            lambda_n = kwargs.pop('lambdas')
            nu_n = kwargs.pop('nus')

            G = series(lambda_n, nu_n, idx0, t, **kwargs)

    return G


def diracdelta(x, idx0, peak=math.inf):
    """Diract delta function."""

    y = np.zeros(x.size)  # zero everywhere ...
    y[idx0] = peak        # ... except at x0
    return y


def gaussian(x, idx0, D0, t):
    """Gaussian normal distribution."""

    MU = x[idx0]
    SIGMA = np.sqrt(2*D0*t)
    y = 1/(SIGMA*np.sqrt(2*np.pi)) * np.exp(-1/2*((x-MU)/SIGMA)**2)
    return y


def series(lambdas, nus, idx0, t_n, sumtype='default', clip=False, thresh=0, **kwargs):
    """Series solution.

    lambdas: (1xN) array
    nus: (NxM) matrix
    idx0: (scalar) index
    t_n: (scalar) time
    sumtype: (str) 'default' or 'fejer' summation
    clip: (bool) whether to clip all values <=thresh to 0
    thresh: (scalar) where to clip
    """

    lambdas = np.squeeze(lambdas)

    nu_0 = nus[:, idx0]
    solution = np.exp(-lambdas[:, np.newaxis] * t_n) * nus * nu_0[:, np.newaxis]

    # sum down the columns over all terms
    if sumtype == 'default':
        y = np.sum(solution, 0)  # standard fourier summation
    elif sumtype == 'fejer':  # fejer summation
        Nmodes = solution.shape[0]
        ps = np.cumsum(solution, 0)/Nmodes  # running average
        y = np.sum(ps, 0)  # sum the partial sums
    else:
        raise Exception('Wrong sumtype')

    if clip:
        y = np.where(y<thresh, 0, y)

    return y
