"""Definition of the analytical solution."""

from math import sqrt, isinf, sin, cos

import numpy as np

from .rootfinding import find_roots


class Solver:

    def __init__(self, diffusivities, lengths, permeabilities):

        # store as numpy array
        self.D = np.array(diffusivities)
        self.L = np.array(lengths)
        self.P = np.array(permeabilities)

        self.eigV = None

    def eval_F(self, x):
        """Evaluate the function F(x)."""
        # no requirements on x, will evaluate any shape and value
        evaluate = np.vectorize(lambda xi: F(xi, K=self.P, D=self.D, L=self.L), otypes=[float])
        return evaluate(x)

    def find_eigV(self, xrange=(0, 1000)):
        """Find eigenvalues."""

        # range
        min_lambda = xrange[0]
        max_lambda = xrange[1]

        # input validation
        if min_lambda > max_lambda:
            raise ValueError('min>max')
        if min_lambda < 0:
            raise ValueError('min<0')

        # calc
        roots, error = find_roots(self.eval_F, xrange)
        self.eigV = roots
        return roots, error


def left_BC(sqrt_lambda, sqrt_D_0, K_L):
    """Apply boundary condition at the left edge of the domain."""

    # initialise
    y = [None, None]

    # apply BCs
    if isinf(K_L):  # Dirichlet condition
        y[0] = 0
        y[1] = 1
    else:  # K_L is real - Neumann condition
        y[0] = sqrt_lambda*sqrt_D_0
        y[1] = K_L

    # done
    return y


def right_BC(y, sqrt_lambda, sqrt_D_m, l_m, K_R):
    """Apply boundary condition at the right edge of the domain."""

    # pre-compute
    val = sqrt_lambda/sqrt_D_m*l_m
    sin_val = sin(val)
    cos_val = cos(val)

    # apply BCs
    if isinf(K_R):  # Dirichlet condition
        F = cos_val*y[0] + sin_val*y[1]
    else:  # K_R is real - Neumann condition
        F = ( K_R*(cos_val*y[0] + sin_val*y[1])
            + sqrt_lambda*sqrt_D_m*(-sin_val*y[0] + cos_val*y[1]) )

    # done
    return F


def internal_BC(y, sqrt_lambda, sqrt_D_i, sqrt_D_ip1, l_i, perm_i):
    """Apply internal boundary condition linking the sub-domains."""

    # pre-compute
    val = l_i*sqrt_lambda/sqrt_D_i
    sin_val = sin(val)
    cos_val = cos(val)
    rlD = (1/perm_i)*sqrt_lambda*sqrt_D_i

    y_old = [y[0], y[1]]  # copy, not ref

    # apply BCs
    y[0] = (cos_val-(sin_val*rlD))*y_old[0] \
         + (sin_val+(cos_val*rlD))*y_old[1]
    y[1] = sqrt_D_i/sqrt_D_ip1 * (-sin_val*y_old[0] + cos_val*y_old[1])

    # done
    return y


def F(eta, K, D, L):
    """Function definition of F(eta).

    eta := (1) eta, x-variable of F
    K := (N+1) membrane permeabilities (including domain end)
    D := (N) compartment diffusivities
    L := (N) compartment lengths
    """

    # pre-calc
    sqrt_lambda = sqrt(eta)
    nCompartments = len(D)

    y = [0, 0]
    y = left_BC(sqrt_lambda, sqrt(D[0]), K[0])
    for i in range(nCompartments-1):
        y = internal_BC(y, sqrt_lambda, sqrt(D[i]), sqrt(D[i+1]), L[i], K[i+1])
    Fval = right_BC(y, sqrt_lambda, sqrt(D[nCompartments-1]), L[nCompartments-1], K[nCompartments])

    return Fval
