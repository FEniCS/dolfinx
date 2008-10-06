# This script demonstrate how to project and interpolate functions
# between different finite element spaces.

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2008-10-06 -- 2008-10-06"
__copyright__ = "Copyright (C) 2008 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

class MyFunction(Function):
    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)

    def eval(self, values, x):
        values[0] = sin(10.0*x[0])*sin(10.0*x[1])

# Create mesh
mesh = UnitSquare(5, 5)

# Create elements
P1 = FiniteElement("Lagrange", "triangle", 1)
P2 = FiniteElement("Lagrange", "triangle", 2)

# Create function
v = MyFunction(P2, mesh)

# Compute projection (L2-projection)
Pv = project(v, P1)

# Compute interpolation (evaluating dofs)
Piv = interpolate(v, P1)

# Plot functions
plot(v,   title="v")
plot(Pv,  title="Pv")
plot(Piv, title="Pi v")
interactive()
