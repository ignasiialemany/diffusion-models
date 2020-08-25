"""Definition of the analytical solution."""

from math import sqrt, isinf, sin, cos
from functools import partial

import numpy as np

from .rootfinding import find_roots
from .diffusion import solution_1D


class Solver:

    def __init__(self, domain):

        # store domain object
        self.domain = domain

        self.eigV = None
        self.eigM = None

    def eval_F(self, x):
        """Evaluate the function F(x)."""
        # no requirements on x, will evaluate any shape and value
        fun = partial(F, domain=self.domain)  # is now fun(x)
        evaluate = np.vectorize(fun, otypes=[float])
        return evaluate(x)

    def find_eigV(self, max_lambda, zero=0, return_error=False, **root_kwargs):
        """Find eigenvalues."""

        # input validation
        if max_lambda < zero:
            raise ValueError('max_lambda < zero')
        if zero < 0:
            raise ValueError('zero < 0')

        # calc
        roots, error = find_roots(self.eval_F, (zero, max_lambda), **root_kwargs)
        if roots[0] != 0:  # roots are sorted, so first one should be zero
            roots = np.insert(roots, 0, 0)  # zero is always a root
        self.eigV = roots
        if return_error:
            return roots, error
        else:
            return roots

    def find_eigM(self, xq, eigV=None):
        """Find eigenmodes for each eigenvalue.

        eigV := (N) eigenvalues
        xq := (M) query points
        -> (N,M) eigenmodes
        """

        if eigV is None:  # not provided
            if self.eigV is None:
                raise Exception('Need to call find_eigV first!')
            eigV = self.eigV  # load pre-computed eigV

        # convert to numpy array
        eigV = np.array(eigV)
        xq = np.array(xq)

        # calculate
        mode = partial(compute_mode, x=xq, domain=self.domain)  # is now mode(eVal)
        eigM = np.array([mode(ev) for ev in eigV])
        self.eigM = eigM
        return eigM

    def verify_eigM(self, eigM=None, pos_dim=1):
        """Check if there are missing or duplicate eigenvalues/eigenmodes.
        Returns the number of sign changes for each eigenmode, and the
        mode number (should match number of sign changes for that mode).

        eigM : (NxM) array of eigenmodes with N eigenvalues and M x-positions
        pos_dim : (scalar) use if eigM is passed transposed
        """

        if eigM is None:
            if self.eigM is None:
                raise Exception('Need to call find_eigM first!')
            eigM = self.eigM

        sign_eigM = np.sign(eigM)  # -1/+1, or 0 for +/- 0
        hasSignChange = np.diff(sign_eigM, n=1, axis=pos_dim) != 0
        nSignChanges = np.sum(hasSignChange, axis=pos_dim)
        nthMode = np.arange(0, len(eigM))
        return nSignChanges, nthMode

    def solve(self, t, xq, idx0, eigV=None, eigM=None):
        """Wrapper to calculate the diffusion solution."""
        eigV = eigV if eigV is not None else self.eigV
        eigM = eigM if eigM is not None else self.eigM
        return solution_1D(t, xq, idx0, 'arbitrary', lambdas=eigV, nus=eigM)


def compute_mode(lambda_, x, domain):
    """Compute the mode corresponding to the eigenvalue.

    lambda_ := (1) lambda, eigenvalue
    x := (M) query points at which to compute the modes
    domain := Domain object with (N) compartments and (N+1) barriers
    """

    # extract shorthand for domain parameters
    L, D, K = domain.lengths, domain.diffusivities, domain.permeabilities

    # pre-compute
    sqrt_lambda = np.sqrt(lambda_)
    sqrt_D = np.sqrt(D)
    r = 1/K[1:-1]  # internal resistance
    barriers = domain.barriers

    # initialise at left boundary
    K_L = K[0]  # rename
    if np.isinf(K_L):  # infinite relaxivity
        V = [0, 1]
    elif K_L == 0:  # no relaxivity
        V = [1, 0]
    else:  # finite relaxivity
        V = [1, K_L/sqrt_lambda]

    y = np.zeros(x.size)
    N = L.size

    indices = domain.locate(x)  # compartment index for each x location
    for i in range(N):  # for each compartment

        idxs = indices == i  # inside this compartment
        val_ra = sqrt_lambda/sqrt_D[i] * (x[idxs]-barriers[i])
        y[idxs] = np.cos(val_ra)*V[0] \
                + np.sin(val_ra)*V[1]

        if i < N-1:  # except last compartment
            val_k = sqrt_lambda*L[i]/sqrt_D[i]
            R_k = [[ np.cos(val_k), np.sin(val_k)], \
                   [-np.sin(val_k), np.cos(val_k)]]
            K_k_kp1 = [[1, sqrt_lambda*sqrt_D[i]*r[i]], \
                       [0, sqrt_D[i]/sqrt_D[i+1]]]
            M_k_kp1 = np.matmul(K_k_kp1, R_k)
            V = np.matmul(M_k_kp1, V)

    # normalise the eigenmode
    Norm = np.sqrt(np.trapz(y**2, x))  # accounts for variable x spacing
    eigenMode = y/Norm

    # Correct last value that is zero by symmetry
    eigenMode[-1] = eigenMode[-2]
    return eigenMode


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


def right_BC(y, sqrt_lambda, sqrt_D_m, L_m, K_R):
    """Apply boundary condition at the right edge of the domain."""

    # pre-compute
    val = sqrt_lambda/sqrt_D_m*L_m
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


def internal_BC(y, sqrt_lambda, sqrt_D_i, sqrt_D_ip1, L_i, K_i):
    """Apply internal boundary condition linking the sub-domains."""

    # pre-compute
    val = L_i*sqrt_lambda/sqrt_D_i
    sin_val = sin(val)
    cos_val = cos(val)
    rlD = (1/K_i)*sqrt_lambda*sqrt_D_i

    y_old = [y[0], y[1]]  # copy, not ref

    # apply BCs
    y[0] = (cos_val-(sin_val*rlD))*y_old[0] \
         + (sin_val+(cos_val*rlD))*y_old[1]
    y[1] = sqrt_D_i/sqrt_D_ip1 * (-sin_val*y_old[0] + cos_val*y_old[1])

    # done
    return y


def F(lambda_, domain):
    """Function definition of F(lambda).

    lambda_ := (1) lambda, x-variable of F
    domain := Domain object with (N) compartments and (N+1) barriers
    """

    # extract shorthand for domain parameters
    L, D, K = domain.lengths, domain.diffusivities, domain.permeabilities

    # pre-calc
    sqrt_lambda = sqrt(lambda_)
    nCompartments = domain.N

    y = [0, 0]
    y = left_BC(sqrt_lambda, sqrt(D[0]), K[0])
    for i in range(nCompartments-1):
        y = internal_BC(y, sqrt_lambda, sqrt(D[i]), sqrt(D[i+1]), L[i], K[i+1])
    Fval = right_BC(y, sqrt_lambda, sqrt(D[nCompartments-1]), L[nCompartments-1], K[nCompartments])

    return Fval
