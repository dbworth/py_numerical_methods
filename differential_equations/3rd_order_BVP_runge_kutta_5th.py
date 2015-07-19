"""
Solving 3rd order DE Boundary Value Problem 
using adaptive 5th order Runge-Kutta

David Butterworth
University of Queensland
"""


import math
import numpy as np
import pylab # plot

#------------------------------------------------------------------------------#

def integrate_rk5(F,x,y,xStop,h,tol=1.0e-6):
    """
    5th order Runge-Kutta
    Calculate the next step of F(x,y)
    """

    def run_kut5(F,x,y,h):

        # Runge-Kutta-Fehlberg formulas
        C = np.array([37./378, 0., 250./621, 125./594, 0., 512./1771])
        D = np.array([2825./27648, 0., 18575./48384, 13525./55296, 277./14336, 1./4])
        n = len(y)
        K = np.zeros((6,n),dtype=np.float64)

        K[0] = h*F(x,y)
        K[1] = h*F(x + 1./5*h, y + 1./5*K[0])
        K[2] = h*F(x + 3./10*h, y + 3./40*K[0] + 9./40*K[1])
        K[3] = h*F(x + 3./5*h, y + 3./10*K[0]- 9./10*K[1] + 6./5*K[2])
        K[4] = h*F(x + h, y - 11./54*K[0] + 5./2*K[1] - 70./27*K[2] + 35./27*K[3])
        K[5] = h*F(x + 7./8*h, y + 1631./55296*K[0] + 175./512*K[1] + 575./13824*K[2] + 44275./110592*K[3] + 253./4096*K[4])

        # Initialize arrays {dy} and {E}
        E = np.zeros((n),dtype=np.float64)
        dy = np.zeros((n),dtype=np.float64)

        # Compute solution increment {dy} and per-step error {E}
        for i in range(6):
            dy = dy + C[i]*K[i]
            E = E + (C[i] - D[i])*K[i]

        # Compute RMS error e
        e = math.sqrt(sum(E**2)/n)
        return dy,e

    X = []
    Y = []
    X.append(x)
    Y.append(y)
    stopper = 0 # Integration stopper(0 = off, 1 = on)

    for i in range(10000):
        dy,e = run_kut5(F,x,y,h)

        # Accept integration step if error e is within tolerance
        if e <= tol:
            y = y + dy
            x = x + h
            X.append(x)
            Y.append(y)

            # Stop if end of integration range is reached
            if stopper == 1: break

        # Compute next step size from Eq. (7.24)
        if e != 0.0:
            hNext = 0.9*h*(tol/e)**0.2
        else: 
            hNext = h

        # Check if next step is the last one; is so, adjust h
        if (h > 0.0) == ((x + hNext) >= xStop):
            hNext = xStop - x
            stopper = 1

        h = hNext

    return np.array(X),np.array(Y)

#------------------------------------------------------------------------------#

# The second-order differential equation
# written as system of first-order DEs
# of the form dy/dx
# so if dx/dt, then it's F(t,x)
def F(x,y):
    f = np.zeros( (3) , dtype=np.float64) # default float64
    f[0] = y[1]
    f[1] = y[2]
    f[2] = 2.0*y[2] + 6.0*x*y[0]

    return f

#------------------------------------------------------------------------------#

# initial values of y,y',y''
#  u is unknown
def initCond(u):
    return np.array( [0.0,0.0,u] )

# boundary condition residual
def r(u):
    print "Called r()"
    print "initCond(u) = ", initCond(u)
    X,Y = integrate(RK4, F, t0, initCond(u), t_end, h)
    #X,Y = integrate(F,xStart, initCond, xStop,h)
    y = Y[len(Y) - 1]
    r = y[0] - 2.0
    return r

# Do the integration
t0    = 5.0 # x_start, starting point for integration
t_end = 0.0 # end point
h = -0.1 # step size
init_cond = np.array( [0.0,0.0,3.] )

t2,X2 = integrate_rk5(F, t0, init_cond, t_end, h)

# in the output, X2[:,0] = first column = y
#                X2[:,1] = second column = y'
print t2
print X2

# Plot
pylab.plot(t2, X2[:,0])
pylab.axis( [-0.1, 5.0, -2.0, 7.0] ) # Set the x,y axis range
pylab.grid()
pylab.show()

