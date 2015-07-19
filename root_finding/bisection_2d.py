"""
2-dimensional Bisection Method

David Butterworth
University of Queensland
"""


import math
import numpy as np
import pylab # plot

#------------------------------------------------------------------------------#

def bisect(f, x1, x2, tol=1.0e-9):
    """
    Locate a root of f(x) by subdividing the original range.

    Input:
        f: user-supplied function f(x)
        x1: first end of range
        x2: other end of range
    Returns:
        x, a point near the root
    """
    assert callable(f), "User-supplied function must be callable."
    assert x1 != x2, "Bad initial range given to bracket."
    f1 = f(x1)
    f2 = f(x2)
    assert f1 * f2 < 0.0, "Range does not clearly bracket a root."
    while abs(x2 - x1) > tol:
        x_mid = 0.5*(x1+x2)
        f_mid = f(x_mid)
        if f_mid == 0.0:
            return x_mid # found it
        if f_mid * f1 < 0.0:
            x2 = x_mid
            f2 = f_mid
        else:
            x1 = x_mid
            f1 = f_mid
    return x_mid


def bisect2d(f, x1, x2, tol=1.0e-9):
    """
    Locate a root of f(x) by subdividing the original range.

    Input:
        f: user-supplied function f(x)
        x1: first end of range
        x2: other end of range
    Returns:
        x, a point near the root
    """
    assert callable(f), "User-supplied function must be callable."
    assert x1 != x2, "Bad initial range given to bracket."
    f1 = f(x1)
    f2 = f(x2)
    assert f1 * f2 < 0.0, "Range does not clearly bracket a root."
    while abs(x2 - x1) > tol:
        x_mid = 0.5*(x1+x2)
        f_mid = f(x_mid)
        if f_mid == 0.0:
            return x_mid # found it
        if f_mid * f1 < 0.0:
            x2 = x_mid
            f2 = f_mid
        else:
            x1 = x_mid
            f1 = f_mid
    return x_mid

#------------------------------------------------------------------------------#

# Bisection should produce a root at (r2,r3).
# Output:
# root (+0.707031,+0.577393)
# func (-0.000076,+0.000042)
r2 = 1.0/math.sqrt(2.0)
r3 = 1.0/math.sqrt(3.0)

def F(x,y):
    return x - r2
def G(x,y):
    return y - r3

x = 0.0; y = 0.0
print "F(x,y) =", F(x,y)
print "G(x,y) =", G(x,y)

print ""

x = 1.0; y = 1.0
print "F(x,y) =", F(x,y)
print "G(x,y) =", G(x,y)


