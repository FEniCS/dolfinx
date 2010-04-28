__author__ = "Marie E. Rognes (meg@simula.no)"
__date__ = "2010-04-28"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU LGPL Version 2.1"

# Last changed: 2010-04-28

from dolfin import *
from time import *

set_log_level(INFO)

# Create mesh and function spaces
n = 4
mesh = UnitCube(n, n, n)

k = 1
V = VectorFunctionSpace(mesh, "CG", k+1)
Q = FunctionSpace(mesh, "CG", k)
W = V * Q

V2 = VectorFunctionSpace(mesh, "CG", k+2)
Q2 = FunctionSpace(mesh, "CG", k+1)
W2 = V2 * Q2

# Create exact dual
dual = Expression(("sin(5.0*x[0])*sin(5.0*x[1])", "x[1]", "1.0",
                   "pow(x[0], 3)"), degree=3)

# Create P1 approximation of exact dual
z1 = Function(W)
z1.interpolate(dual)

# Create P2 approximation from P1 approximation
tic = time()
z2 = Function(W2)
z2.extrapolate(z1)
elapsed = time() - tic;

print "\nTime elapsed: %0.3g s." % elapsed
