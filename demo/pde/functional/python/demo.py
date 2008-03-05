# This demo program computes the value of the functional
#
#     M(v) = int v^2 + (grad v)^2 dx
#
# on the unit square for v = sin(x) + cos(y). The exact
# value of the functional is M(v) = 2 + 2*sin(1)*(1-cos(1))
#
# The functional M corresponds to the energy norm for a
# simple reaction-diffusion equation.
#
# Original implementation: ../cpp/main.cpp by Anders Logg
#
__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-14 -- 2007-11-28"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Create mesh and finite element
mesh = UnitSquare(16, 16)
element = FiniteElement("Lagrange", "triangle", 2)

# The function v
class MyFunction(Function):
    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)
    
    def eval(self, values, x):
        values[0] = sin(x[0]) + cos(x[1])

# Define variational problem
# Test and trial functions
v = MyFunction(element, mesh)
M = (v*v + dot(grad(v), grad(v)))*dx

value = assemble(M, mesh)

# Compute exact value
exact_value = 2.0 + 2.0*sin(1.0)*(1.0 - cos(1.0))

print "The energy norm of v is %.15g (should be %.15g)." %(value, exact_value)

