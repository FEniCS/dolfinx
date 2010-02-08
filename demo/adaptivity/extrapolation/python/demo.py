__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2010-02-08"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU LGPL Version 2.1"

# Last changed: 2010-02-08

from dolfin import *

# Create mesh and function spaces
mesh = UnitSquare(8, 8)
P1 = FunctionSpace(mesh, "CG", 1)
P2 = FunctionSpace(mesh, "CG", 2)

# Create exact dual
dual = Expression("sin(5.0*x[0])*sin(5.0*x[1])")

# Create P1 approximation of exact dual
z1 = Function(P1)
z1.interpolate(dual)

# Create P2 approximation from P1 approximation
z2 = Function(P2)
z2.extrapolate(z1)

# Plot approximations
plot(z1, title="z1")
plot(z2, title="z2")
interactive()
