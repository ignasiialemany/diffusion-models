"""Routines for root finding."""

import math
import warnings

import numpy as np
from scipy.optimize import brentq
from chebpy import chebfun
from chebpy.core.settings import userPrefs as ChebpyPrefs


def find_roots(F, x_range, root_accuracy=dict(), equal_tol=dict()):
    """Find roots of function in a given range.

    Fits `chebfun`s to the function and refines it to find roots
    Usage:
        Inputs:
            F        : function handle that evaluates F(x)
            x_range  : [xmin, xmax] range of x over which to search
        Outputs:
            x0  : row vector of x values where F(x) := 0
            err : row vector of F(x) values (== error, since F(x0)===0)
    """

    ## 1) subdivide the function into segments ...
    # In the first iteration, this fits a single global chebfun to the function over the entire domain.
    # Next, we recursively subdivide into segments where we expect roots.
    intervals = subdivide(F, split_into_interval(x_range), equal_tol=equal_tol)  # update intervals

    ## 2) ... and find the roots of the function.
    # Process each interval to find the local zeros
    roots_all = []  # empty
    for a, b in intervals:
        roots = find_roots_in_interval(F, a, b, **root_accuracy)  # always a list
        roots_all.extend(roots)

    ## 3) post-process
    x0 = np.unique(roots_all)  # sorted
    if x0.size > 0:  # if we found any
        err = F(x0)  # F evaluated at x0 corresponds to the error, since F(x0)===0
    else:
        err = None
    return x0, err


def subdivide(F, x_ranges, equal_tol=dict()):
    """Recurisvely split intervals of F into intervals with at most one root."""

    x_ranges_new = np.array([[]])  # array of arrays, shape important for concat later!
    for x_range in x_ranges:  # go over each interval

        # further divide (if necessary)
        extrema = local_extrema(F, x_range, equal_tol=equal_tol)  # find the extrema in this range
        if np.array_equal(x_range, extrema):  # same interval came out as we put in
            x_ranges_i = np.array([x_range])  # done, reshape into column
        else:
            # further divide
            subintervals = split_into_interval(extrema)
            x_ranges_i = subdivide(F, subintervals, equal_tol=equal_tol)  # recurse

        # store
        if x_ranges_i.size:  # found something
            if x_ranges_new.size:  # already has some
                x_ranges_new = np.concatenate((x_ranges_new, x_ranges_i), 0)
            else:  # empty, so initiate with newly found ranges
                x_ranges_new = x_ranges_i

    return x_ranges_new


def equal(a, b, abstol=1e-16, reltol=0, mathlib=math):
    """Check if two values are equal.

    By default, we test with (absolute) tolerances. This can be changed as desired.
    For exact equality, set abstol=reltol=0.

    mathlib allows to specify the module from which isclose should be used: numpy, math, or custom.
    A custom module needs to implement .__name__ and .isclose(a, b, abstol=__, reltol=__)
    """

    # depending on choice of function, assign tolerances
    if mathlib.__name__ == 'math':
        tols = dict(abs_tol=abstol, rel_tol=reltol)
    elif mathlib.__name__ == 'numpy':
        tols = dict(atol=abstol, rtol=reltol)
    else:
        tols = dict(abstol=abstol, reltol=reltol)

    # call function with values
    return mathlib.isclose(a, b, **tols)


def local_extrema(F, x_range, equal_tol=dict()):
    """Find local extrema, including endpoints.
    
    Returns the extrema in a numpy array.
    """

    # check that the interval is not degenerate
    if equal(*x_range, abstol=0, reltol=0):  # exact equality
        return x_range

    # find extrema
    cheb = chebfun(F, x_range)  # automatically constructed
    maxpow2 = ChebpyPrefs.maxpow2
    max_n = 2**(maxpow2-1)  # one exponent less to be safe
    converged = np.all([f.size < max_n for f in cheb.funs])
    if not converged:
        n_new = 1000  # probably good enough
        warnings.warn('chebfun did not converge in [{0:g}, {1:g}].'.format(*x_range)
                     +' Approximating using n={:d}'.format(n_new))
        cheb = chebfun(F, x_range, n=n_new)  # hopefully this is accurate enough
    dcheb = cheb.diff()
    extrema = dcheb.roots()
    extrema = np.sort(extrema)  # sort into numpy array
    extrema = np.clip(extrema, *x_range)  # in case numerical error puts it just outside

    # add end points
    if extrema.size > 0:  # at least one extremum found
        # endpoint a
        if equal(extrema[0], x_range[0], **equal_tol):
            extrema[0] = x_range[0]
        else:
            extrema = np.insert(extrema, 0, x_range[0])
        # endpoint b
        if equal(extrema[-1], x_range[1], **equal_tol) and extrema.size > 1:
            # size check ensures we don't override x_range[0]
            extrema[-1] = x_range[1]
        else:
            extrema = np.insert(extrema, extrema.size, x_range[1])
    else:
        extrema = np.array(x_range)

    # done
    return extrema


def find_roots_in_interval(F, a, b, abstol=1e-22, reltol=1e-10, maxiter=100, warn=True):
    """Find a single root in an interval.
    
    This assumes F only has one root in the inverval (a, b).
    F(a) or F(b) may be zero, in which case only endpoint roots are returned.
    """

    Fa, Fb = F(a), F(b)  # values on interval end
    if Fa * Fb > 0:  # same side of x-axis and neither endpoint==0
        return []  # no root here
    elif Fa==0 or Fb==0:  # zero(s) directly on interval boundary (unlikely)
        return [x for x, y in zip([a, b],[Fa, Fb]) if y==0]
    else:  # there has to be a root since yA and yB have different sign
        # https://mathworks.com/help/matlab/ref/fzero.html
        fsolve = brentq  # for single root, very robust & fast
        tols = dict(xtol=abstol, rtol=reltol, maxiter=maxiter)
        root, result = fsolve(F, a, b, full_output=True, disp=False, **tols)
        if not result.converged and warn:
            warnings.warn("fsolve did not converge in [{0:g},{1:g}]".format(a,b))
        return [root]


def split_into_interval(x):
    """Divide points into intervals."""

    intervals = np.column_stack((x[:-1], x[1:]))  # two columns
    return intervals
