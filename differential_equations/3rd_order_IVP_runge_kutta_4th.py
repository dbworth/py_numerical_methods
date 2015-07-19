"""
Solving 3rd order ODE Initial Value Problem 
using 2nd order (modified Euler) and 4th order Runge-Kutta

David Butterworth
University of Queensland
"""


import math
import numpy as np
import pylab # plot

#------------------------------------------------------------------------------#

def fn(t): 
    """
    Analytical solution to the differential equation
    """
    # must use exp() from numpy to handle input from linspace
    return 0.1*(-3.0*np.exp(-t) + np.exp(2.0*t)*np.sin(t) + 3.0*np.exp(2.0*t)*np.cos(t) + 2.0)

#------------------------------------------------------------------------------#

def RK2(F,x,y,h):
    """
    2nd order Runge-Kutta (modified Euler)
    Calculate the next step of F(x,y)
    """
    return h * ( F(x,y) + F(x + h, y + h*F(x,y)) ) / 2.0


def RK4(F,x,y,h):
    """
    4th order Runge-Kutta
    Calculate the next step of F(x,y)
    """
    k1 = h * F(x,y)
    k2 = h * F(x + h/2.0, y + k1/2.0)
    k3 = h * F(x + h/2.0, y + k2/2.0)
    k4 = h * F(x + h, y + k3)
    return (k1 + 2*k2 + 2*k3 + k4) / 6.0


def integrate(method,F,x,y,xStop,h):
    """
    Numerically integrate a first order differential equation, 
    or an array of DEs.

    This wraps the specified method e.g. RK2, RK4

    For a 2nd order DE, the output is for example: t,[y0,y1]
    """
    X = []
    Y = []
    X.append(x)
    Y.append(y)
    while (x < xStop):
        h = min(h,xStop - x)
        y = y + method(F,x,y,h)
        x = x + h
        X.append(x)
        Y.append(y)
    return np.array(X),np.array(Y)

#------------------------------------------------------------------------------#

# The second-order differential equation
# written as system of first-order DEs
# of the form dy/dx
# so if dx/dt, then it's F(t,x)
def F(x,y):
    print "called F() with x =", x
    f = np.zeros( (3) ) # default float64
    f[0] = y[1]
    f[1] = y[2]
    f[2] = 1.0 + 3.0*y[2] - y[1] - 5*y[0] 
    return f

#------------------------------------------------------------------------------#

# Do the integration
t0    = 0.0 # starting point for integration
t_end = 3.0 # end point
h = 0.1 # step size
init_cond = np.array([1.0/5.0,1.0,1.0]) # initial conditions [y0,y1,y2]
t1,X1 = integrate(RK2, F, t0, init_cond, t_end, h)
t2,X2 = integrate(RK4, F, t0, init_cond, t_end, h)
# in the output, X1[:,0] = first column = y
#                X1[:,1] = second column = y'

# Plot:
at = np.linspace(0.0, 10.0, 1000)
ax = fn(at)

pylab.plot(t1, X1[:,0])  # Modified Euler (rk2) h=0.1
pylab.plot(t2, X2[:,0])  # 4th order Runge-Kutta (rk4) h=0.1
pylab.plot(at, ax, 'r--'  ) # plot analytical solution
pylab.axis( [-0.1, 3.0, -3.0, 3.0] ) # Set the x,y axis range
pylab.grid()
pylab.show()

