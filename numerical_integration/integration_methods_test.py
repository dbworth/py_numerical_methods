"""
Numerical integration test

A comparison of 5 methods:
 - Trapezoid rule
 - Romberg integration
 - Adaptive Simpsons rule
 - Gauss-Legendre m-point quadrature
 - Adaptive quadrature by comparing two integration rules

Usage:
python ./test_integration_methods.py

David Butterworth
University of Queensland

Based on examples from
"Numerical Methods in Engineering with Python" by Jaan Kiusalaas
& http://www.csee.umbc.edu/~squire/cs455_l8.html
"""

import math # sqrt(), cos(), pi
import numpy as np # zeros, float64
import pylab # plot

#------------------------------------------------------------------------------#


def trapezoid(f,a,b,Iold,k):
    """
    Recursive Trapezoidal Rule

    A Newton-Cotes type method, having equally spaced sample points.
    
    The area under the function is calculated by finding the area of
    1 trapzoid, this is then divided into 2 trapezoid panels, which
    can then be divided into 4 panels, and so on... until an accurate
    answer is achieved.

    No. of panels = 2^(k-1)
    Iold = the previously calculated area for k-1 panels.
    This function outputs the next sumation being for k panels.
    """
    if k == 1: # k=1
        Inew = (f(a) + f(b))*(b - a)/2.0 # (1/2)*base*height
    else: # k=2,3,4...
        n = 2**(k-2) # Number of new points
        h = (b - a)/n # Spacing of new points
        x = a + h/2.0 # Coord. of 1st new point
        sum = 0.0
        for i in range(n):
            sum = sum + f(x)
            x = x + h
            Inew = (Iold + h*sum)/2.0
    return Inew


#------------------------------------------------------------------------------#


def romberg(f,a,b,tol=1.0e-6):
    """
    Romberg Integration 
    of f(x) from x = a to b.

    usage:
    I,nPanels = romberg(f,a,b,tol=1.0e-6)

    An efficient extension to the Trapezoid Rule, using Richardson Extrapolation

    Returns integral, no. of panels
    """
    def richardson(r,k):
        for j in range(k-1,0,-1):
            const = 4.0**(k-j)
            r[j] = (const*r[j+1] - r[j])/(const - 1.0)
        return r

    r = np.zeros(21, dtype=np.float64)
    r[1] = trapezoid(f,a,b,0.0,1)
    r_old = r[1]

    for k in range(2,21):
        r[k] = trapezoid(f,a,b,r[k-1],k)
        r = richardson(r,k)

        if abs(r[1]-r_old) < tol*max(abs(r[1]),1.0):
            return r[1],2**(k-1)

        r_old = r[1]

    print "Romberg quadrature did not converge"


#------------------------------------------------------------------------------#


n_simp = 1 # global
def adaptive_simpson( f, a, b, tol=1.0e-6):
    """
    Adaptive Simpsons Rule

    Evaluates the integral of f(x) on [a,b].

    This algorithm is recursive but not efficient.
    """

    tol_factor = 10.0 # more conservative than normal factor of 15

    h = 0.5 * ( b - a )

    x0 = a
    x1 = a + 0.5 * h
    x2 = a + h
    x3 = a + 1.5 * h
    x4 = b

    f0 = f( x0 )
    f1 = f( x1 )
    f2 = f( x2 )
    f3 = f( x3 )
    f4 = f( x4 )

    s0 = h * ( f0 + 4.0 * f2 + f4 ) / 3.0
    s  = h * ( f0 + 4.0 * f1 + 2.0 * f2 + 4.0 * f3 + f4 ) / 6.0

    global n_simp

    if abs( s0 - s ) >= tol_factor * tol:
        s = adaptive_simpson( f, x0, x2, 0.5 * tol ) + \
            adaptive_simpson( f, x2, x4, 0.5 * tol )
        n_simp += 1 # increase iteration count

    return s


#------------------------------------------------------------------------------#


def gaussNodes(m,tol=10e-9):
    """
    x,A = gaussNodes(m,tol=10e-9)
    Returns nodal abscissas {x} and weights {A} of
    Gauss-Legendre m-point quadrature.
    """

    def legendre(t,m):
        p0 = 1.0
        p1 = t

        for k in range(1,m):
            p = ((2.0*k + 1.0)*t*p1 - k*p0)/(1.0 + k)
            p0 = p1
            p1 = p

        dp = m*(p0 - t*p1)/(1.0 - t**2)
        return p,dp

    A = np.zeros(m,dtype=np.float64)
    x = np.zeros(m,dtype=np.float64)
    nRoots = (m + 1)/2 # Number of non-neg. roots

    for i in range(nRoots):
        t = math.cos(math.pi*(i + 0.75)/(m + 0.5)) # Approx. root

        for j in range(30):
            p,dp = legendre(t,m) # Newton-Raphson
            dt = -p/dp
            t = t + dt # method

            if abs(dt) < tol:
                x[i] = t; x[m-i-1] = -t
                A[i] = 2.0/(1.0 - t**2)/(dp**2) # Eq.(6.25)
                A[m-i-1] = A[i]
                break

    return x,A

def gaussQuad(f,a,b,m):
    """
    I = gaussQuad(f,a,b,m).
    Computes the integral of f(x) from x = a to b
    226 Numerical Integration
    with Gauss-Legendre quadrature using m nodes.
    """
    c1 = (b + a)/2.0
    c2 = (b - a)/2.0
    x,A = gaussNodes(m)
    #print "qauss nodes x = ", x
    sum = 0.0

    for i in range(len(x)):
        sum = sum + A[i]*f(c1 + c2*x[i])

    return c2*sum


#------------------------------------------------------------------------------#


aquad_samples = np.array([])

stack = [[0,1,2,3,4,5,6]]   # stack to store/retrieve 

def store(s0, s1, s2, s3, s4, s5, s6):
    a = [s0, s1, s2, s3, s4, s5, s6]
    stack.append(a)

def retrieve():
    a = stack.pop()
    return a[0], a[1], a[2], a[3], a[4], a[5], a[6]

# Integration method 1
def Sn(F0, F1, F2, h):
    return h*(F0 + 4.0*F1 + F2)/3.0

# Integration method 2
def RS(F0, F1, F2, F3, F4, h):
    return h*(14.0*F0 +64.0*F1 + 24.0*F2 + 64.0*F3 + 14.0*F4)/45.0 
    # error term  8/945  h^7 f^(8)(c)

def aquad3(f, xmin, xmax, eps):
    """
    Adpative quadrature
    """
    top = 0
    value = 0.0
    tol = eps
    ns = 32
    a = xmin
    hs = (xmax-xmin)/ns
    b = a + hs
    stack.pop() # get rid of initial junk

    for i in range(ns): # hueristic starter set
        h1 = (b-a)/2.0
        c = a + h1
        Fa = f(a)
        Fc = f(c)
        Fb = f(b)
        Sab = Sn(Fa, Fc, Fb, h1)
        store(a, Fa, Fc, Fb, h1, tol, Sab)
        top += 1
        #print "top"
        a = b
        b = a + hs

    global aquad_samples

    while(top > 0):
        top -= 1
        a, Fa, Fc, Fb, h1, tol, Sab = retrieve()
        #print "a = ",a
        aquad_samples = np.append(aquad_samples, a)
        c = a + h1
        b = a + 2.0*h1
        h2 = h1/2
        d = a + h2
        e = a + 3.0*h2
        Fd = f(d)
        Fe = f(e)
        Sac = Sn(Fa, Fd, Fc, h2)
        Scb = Sn(Fc, Fe, Fb, h2)
        S2ab = Sac + Scb

        if abs(S2ab-Sab) < tol or h2 < 1.0e-13:
            val = RS(Fa, Fd, Fc, Fe, Fb, h2)
            value += val

        else:
            h1 = h2
            tol = tol/2.0
            store(a, Fa, Fd, Fc, h1, tol, Sac)
            top += 1
            #print "top"
            store(c, Fc, Fe, Fb, h1, tol, Scb)
            top += 1
            #print "top"

    return value


#------------------------------------------------------------------------------#


# Sample functions:

# sqrt(x)*cos(x), over 0 to PI
def f(x): 
    return math.sqrt(x)*math.cos(x)

# same function after change of variables, over 0 to sqrt(PI)
def f2(x): 
    return 2.0*(x**2)*cos(x**2)

# (sin(x)/x)^2
def f3(x): 
    return (math.sin(x)/x)**2


#------------------------------------------------------------------------------#


# Trapezoid Rule with max 2^30 panels
Iold = 0.0
for k in range(1,21):
    Inew = trapezoid(f, 0.0, math.pi, Iold, k )
    #Inew = trapezoid(f2,0.0,math.sqrt(math.pi),Iold,k)
    if (k > 1) and (abs(Inew - Iold)) < 1.0e-6: # stop if error gets small
        break
    Iold = Inew

print "\nBy Trapezoid Rule: "
print "Integral = ", Inew
print "Used ", 2**(k-1), "panels, out of max ", 2**20, " panels."
print "\n"
#raw_input("\nPress return to exit")


# Romberg Integration
# (this converges with much less panels)
print "\nBy Romberg Integration:"
I,n = romberg(f, 0.0, math.pi)
#I,n = romberg(f,0,math.sqrt(math.pi))
print "Integral = ", I
print "Used ", n, "panels, out of max ", 2**20, " panels."
print "\n"


# Adaptive Simpsons rule
print "\nBy Adaptive Simpsons Rule:"
s = adaptive_simpson(f, 0.0, math.pi)
print "Integral = ", s
print "with ", 2**(n_simp-1), "panels"
print "\n"



# Gaussian quadrature
a = 0.0
b = math.pi;
#Iexact = 1.41815
I_old = 0.0
for m in range(2,50):
    I = gaussQuad(f,a,b,m)
    #if abs(I - Iexact) < 0.00001:
    if abs(I - I_old) < 1.0e-6:
        print "\nBy Gaussian Quadrature:"
        print "Number of nodes =",m
        print "Integral =", gaussQuad(f,a,b,m)
        print "\n"
        break
    else:
        I_old = I
        if (m==50):
            print "Gaussian Quadrature failed to converge with", m, "nodes"


# Adaptive quadrature
xmin = 0.0
xmax = math.pi
eps  = 1.0e-6
print "By adaptive quadrature:"
area = aquad3(f, xmin, xmax, eps)
print "Integral =", area
aquad_samples = np.append(aquad_samples, math.pi) # the endpoint is left off the list
#print "samples: ", aquad_samples
print "Numer of samples: ", len(aquad_samples)
print "\n\n"

## Show plot of adaptive quadrature
x = np.linspace(0, math.pi, 200)
y = np.zeros( len(x) )
for i in range(len(x)):
    y[i] = f(x[i])
pylab.plot(x, y) # 'rx'
for s in aquad_samples:
    pylab.plot( [s,s] , [-2.0,f(s)], 'b' )
pylab.show()




