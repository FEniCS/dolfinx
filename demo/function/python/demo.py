__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2008-03-17 -- 2008-03-17"
__copyright__ = "Copyright (C) 2008 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *
from numpy import array

class F(Function):
    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)

    def eval(self, values, x):
        values[0] = sin(3.0*x[0])*sin(3.0*x[1])*sin(3.0*x[2])

# Create mesh and a point in the mesh
mesh = UnitCube(8, 8, 8);
x = array((0.3, 0.3, 0.3))
values = array((0.0,))

# A user-defined function
element = FiniteElement("Lagrange", "tetrahedron", 2)
f = F(element, mesh)

# Project to a discrete function
v = TestFunction(element)
u = TrialFunction(element)
a = v*u*dx
L = v*f*dx
pde = LinearPDE(v*u*dx, v*f*dx, mesh)
g = pde.solve()

# Evaluate user-defined function f
f.eval(values, x)
print "f(x) =", values[0]

# Evaluate discrete function g (projection of f)
g.eval(values, x)
print "g(x) =", values[0]
