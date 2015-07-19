"""
Solve a pair of equations

David Butterworth
University of Queensland
"""


from scipy.optimize import fsolve
import math

#------------------------------------------------------------------------------#

# Test 1
def equations(p):
    x, y = p
    return (x+y**2-4, math.exp(x) + x*y - 3)

x, y =  fsolve(equations, (1, 1))
print x,y
print equations((x, y))
print "\n"

#------------------------------------------------------------------------------#

# Test 2
def equations2(p):
    x, y = p
    return (x - 1.0/math.sqrt(2.0), y - 1.0/math.sqrt(3.0))

x, y =  fsolve(equations2, (1, 1))
print x,y
print equations2((x, y))
print "\n"

# Bisection also finds a root at
# root (+0.707031,+0.577393)
# function value (-0.000076,+0.000042)

