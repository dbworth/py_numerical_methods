"""
Numerical integration test

Gauss quadrature using legendre polynomial weights
and
Adaptive quadrature

David Butterworth
University of Queensland

"""

import numpy as np
import math # sqrt(), cos(), pi

#------------------------------------------------------------------------------#


def integrate(fn, n, a, b, zerodiff=1e-6):
    """
    Applies n-point Gauss Quadrature to the function fn from a to b, using
    Legendre Polynomial weights
    
    Parameters
    ----------
    fn : function
        Integrand - function taking a single argument
    n : string
        Degree of Gaussian Quadrature to apply. Can be one of '2', '3', ... '6'
    a : number
        Lower integral bound
    b : number
        Upper integral bound
    zerodiff : number
        Minimum allowable difference between a and b. Anything less than this
        will return an integral of 0
    
    Returns
    -------
    
    out : number
        Result of applying n-point Guassian Quadrature to fn from a to b
    
    Examples
    --------
    
    >>> integrate(lambda x:1, '2', 4, 5)
    1.0
    
    >>> integrate(lambda x:x**3, '4', -2, 2)
    0.0
    
    """
    
    # First check to see if the bounds we were given make any sense
    if abs(b - a) < zerodiff:
        return 0
    
    # A dict of vectors of gaussian weights
    # NB: We are using the Legendre polynomial weights here
    weights = {}
    
    # A dict of vectors of gussian abscissas (sample points) +-xi
    abscissas = {}
    
    # Weights and Abscissas for 2 point integration...
    # NB: We only store half of the points, as each list is symmetic
    # We construct the full lists later
    weights["2"] = [1]
    abscissas["2"] = [0.5773502691896257]
    
    # 3 point integration...
    weights["3"] = [0.5555555555555556, 0.8888888888888888]
    abscissas["3"] = [0.7745966692414834, 0]
    
    # etc...
    weights["4"] = [0.3478548451374538, 0.6521451548625461]
    abscissas["4"] = [0.8611363115940526, 0.3399810435848563]
    
    weights["5"] = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889]
    abscissas["5"] = [0.9061798459386640, 0.5384693101056831, 0]
    
    weights["6"] = [0.1713244923791704, 0.3607615730481386, 0.4679139345726910]
    abscissas["6"] = [0.9324695142031521, 0.6612093864662645, 0.2386191860831969]
    
    # Normalise the function to the integration bounds [-1, 1]
    # NB: We also need to multiply the result of the integration by (b-a)*0.5
    # This happens laer on
    f_transposed = lambda x: fn( ((b-a)*0.5)*x + ((b+a)*0.5) )
    
    # Now compute the full weight and abscissas lists
    # The logic here is a little complex, but basically expands the above
    # lists (exploiting the symmertry of gauss quadrature points) to include
    # all n points
    n_int = int(n)    
    full_weights = np.zeros(n_int)
    full_abscissas = np.zeros(n_int)
    partial_weights = weights[n]
    partial_abscissas = abscissas[n]
    
    for i in range(n_int):
        # This complex beast calculates the index into the partial weight lists
        partial_index = int(np.floor(
            np.floor((n_int-1)*0.5) - np.abs(i - (n_int-1)*0.5) + 0.5
        ))
        
        # While this expression determines the appropriate sign for this
        # abscissa term
        abscissa_sign = int(np.sign(i - (n_int-1)*0.5))
        
        full_weights[i] = partial_weights[partial_index]
        full_abscissas[i] = partial_abscissas[partial_index] * abscissa_sign
    
    #print "Doing %d point gauss quad from %.4f to %.4f" % (n_int, a, b)
    #print full_weights
    #print full_abscissas
    
    # Compte the quadrature integral
    quad = ((b-a)*0.5) * np.sum(
        np.multiply(
            full_weights,
            map(f_transposed, full_abscissas)
        )
    )
    
    return quad
    

def integrate_adaptive(fn, a, b, tol=1e-6):
    """
    Integrates fn using recursive local adaptive gaussian quadrature from
    a to b
    
    Parameters
    ----------
    fn : function
        Integrand - function taking a single argument
    a : number
        Lower integral bound
    b : number
        Upper integral bound
    tol : number
        Maximum permissible error in integration result
    
    Returns
    -------
    out : number
        Result of intgration
    
    """
    
    # Use 5th and 6th order gauss quadrature
    int_5 = integrate(fn, '5', a, b)
    int_6 = integrate(fn, '6', a, b)
    error = np.abs(int_6 - int_5)
    
    q = int_5
    
    if error > tol:
        m = (a + b) * 0.5
        q = integrate_adaptive(fn, a, m, tol*0.5) +\
            integrate_adaptive(fn, m, b, tol*0.5)
    
    return q

#------------------------------------------------------------------------------#

# Function to integrate:
# sqrt(x)*cos(x), over 0 to PI
fn = lambda x: math.sqrt(x)*math.cos(x)
a = 0.0
b = math.pi

#------------------------------------------------------------------------------#


if __name__ == "__main__":
    
    tolerance = 1e-4
    
    print "\nIntegrating using gaussian quadrature Vs. adaptive gaussian quadrature \n"
    
    print "n\tIntegral\tAdaptive Integral"        
    print "-----------------------------------------"
        
    # Integrate using n-points, where n = 2 to 6
    for j in range(2, 7):
        integral = integrate(fn, "%d" % j, a, b)
        adaptive = integrate_adaptive(fn, a, b)
            
        print "%d\t%.5f\t%.5f" % (j, integral, adaptive)
    print ""

