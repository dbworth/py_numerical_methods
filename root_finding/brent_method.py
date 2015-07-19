"""
Brent's method for root finding

David Butterworth
University of Queensland
"""


import math
import numpy as np
import pylab # plot
import sys # for err()

#------------------------------------------------------------------------------#

def err(string):
    """
    err(string).
    Prints 'string' and terminates program.
    """
    print string
    raw_input('Press return to exit')
    sys.exit()

#------------------------------------------------------------------------------#

def brent(f,a,b,tol=1.0e-9):
    """
    root = brent(f,a,

    Finds root of f(x) = 0 using simplified version of Brent's method
    by combining quadratic interpolation with bisection.
    The root must be bracketed in (a,b).
    Calls user-supplied function f(x).

    From book: "Numerical methods in engineering with Python"
    """
    x1 = a
    x2 = b

    f1 = f(x1)
    if f1 == 0.0: 
        return x1

    f2 = f(x2)
    if f2 == 0.0: 
        return x2

    if f1*f2 > 0.0: 
        err('Root is not bracketed')
        #error.err('Root is not bracketed')

    x3 = 0.5*(a + b)

    for i in range(30):
        f3 = f(x3)

        if abs(f3) < tol: 
            return x3

        # Tighten the brackets of the root
        if f1*f3 < 0.0: 
            b = x3
        else: 
            a = x3

        if (b - a) < tol*max( abs(b),1.0 ): 
            return 0.5*(a + b)

        # Try quadratic interpolation
        denom = (f2 - f1)*(f3 - f1)*(f2 - f3)
        numer = x3*(f1 - f2)*(f2 - f3 + f1) + f2*x1*(f2 - f3) + f1*x2*(f3 - f1)

        # If division by zero, push x out of bounds
        try: 
            dx = f3*numer/denom
        except ZeroDivisionError: 
            dx = b - a

        x = x3 + dx

        # If interpolation goes out of bounds, use bisection
        if (b - x)*(x - a) < 0.0:
            dx = 0.5*(b - a)
            x = a + dx

        # Let x3 <-- x & chose new x1 and x2 so that x1 < x3 < x2
        if x < x3:
            x2 = x3
            f2 = f3
        else:
            x1 = x3
            f1 = f3

        x3 = x

    print 'Too many iterations in brent()'

#------------------------------------------------------------------------------#

if __name__ == '__main__': 

    # Function definitions:
    def f1(x): return x**3.0 - 10.0*x**2.0 + 5.0
    def f2(x): return x * abs( np.cos(x) ) - 1.0

    #print "root=", brent(f1,0.6,0.8)
    print "root=", brent(f1,0.0,8.0)
    #print "root=", brent(f2,2.0,2.2) # tightly constrained boundaries
    print "root=", brent(f2,0.0,4.0) # boundaries outside minima, but it stills finds the root

    # Plot:
    x1 = np.linspace(0.0, 10.0, 1000)
    y1 = f1(x1)

    x2 = np.linspace(0.0, 10.0, 1000)
    y2 = f2(x2)

    pylab.plot(x1, y1)
    pylab.plot(x2, y2)
    pylab.axis( [0.0, 10.0, -150.0, 20.0] ) # Set the x,y axis range
    pylab.grid()
    pylab.show()

