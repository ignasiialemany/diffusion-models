"""Routines for root finding."""

import math
import warnings

import numpy as np
from scipy.optimize import brentq
from chebpy import chebfun


def find_roots(F, xrange):
    """Find roots of function in a given range.

    Fits `chebfun`s to the function and refines it to find roots
    Usage:
        Inputs:
            F       : function handle that evaluates F(x)
            xrange  : [xmin, xmax] range of x over which to search
        Outputs:
            x0  : row vector of x values where F(x) := 0
            err : row vector of F(x) values (== error, since F(x0)===0)
    """

    ## 1) find the extrema of the function, ...
    # We do this by fitting a single global chebfun to the function over the entire domain
    e0_x = local_extrema(F, xrange)

    ## 2) ... subdivide it into segments, ...
    # Recursively subdivide into segments where we expect roots
    intervals = subdivide(F, split_into_interval(e0_x))  # update the intervals

    ## 3)... and find the roots of the function.
    # Process each interval to find the local zeros
    roots_all = np.array([])  # empty
    for a, b in intervals:
        roots = find_roots_in_interval(F, a, b)  # may be []
        roots_all = np.append(roots_all, roots)

    ## 4) post-process
    x0 = np.unique(roots_all)  # sorted
    if x0.size > 0:  # if we found any
        err = F(x0)  # F evaluated at x0 corresponds to the error, since F(x0)===0
    else:
        err = None
    return x0, err


def subdivide(F, xranges):
    """Recurisvely split intervals of F into intervals with at most one root."""

    ranges_new = np.array([[]])  # array of arrays, shape important for concat later!
    for xrange in xranges:  # go over each interval

        # further divide (if necessary)
        extrema = local_extrema(F, xrange)  # find the extrema in this range
        if np.array_equal(xrange, extrema):  # same interval came out as we put in
            xranges_i = np.array([xrange])  # done, reshape into column
        else:
            # further divide
            subintervals = split_into_interval(extrema)
            xranges_i = subdivide(F, subintervals)  # recurse

        # store
        if xranges_i.size:  # found something
            if ranges_new.size:  # already has some
                ranges_new = np.concatenate((ranges_new, xranges_i), 0)
            else:  # empty, so initiate with newly found ranges
                ranges_new = xranges_i

    return ranges_new


def equal(a, b, abstol=1e-20, reltol=1e-9, mathlib=math):
    """Check if two values are equal.

    If exact=True, we do an exact check for equality. This can cause problems!
    If exact=False, we use isclose with abstol=1e-20, reltol=1e-9.

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


def local_extrema(F, xrange):
    """Find local extrema, including endpoints.
    
    Returns the extrema in a numpy array.
    """

    # check that the interval is not degenerate
    if equal(*xrange, abstol=0, reltol=0):  # exact equality
        return xrange

    # find extrema
    cheb = chebfun(F, xrange)  # automatically constructed
    dcheb = cheb.diff()
    extrema = dcheb.roots()
    extrema = np.array(extrema)  # convert

    # add end points
    if extrema.size > 0:  # at least one extremum found
        # endpoint a
        if equal(extrema[0], xrange[0]):
            extrema[0] = xrange[0]
        else:
            extrema = np.insert(extrema, 0, xrange[0])
        # endpoint b
        if equal(extrema[-1], xrange[1]) and extrema.size > 1:
            # size check ensures we keep don't override xrange[0]
            extrema[-1] = xrange[1]
        else:
            extrema = np.insert(extrema, extrema.size, xrange[1])
    else:
        extrema = np.array(xrange)

    # done
    return extrema


def find_roots_in_interval(F, a, b, abstol=1e-22, reltol=1e-10, maxiter=10):
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
        if not result.converged:
            warnings.warn("fsolve did not converge in [{0:g},{1:g}]".format(a,b))
        return root


def split_into_interval(x):
    """Divide points into intervals."""

    intervals = np.column_stack((x[0:-1], x[1:]))  # two columns
    return intervals
