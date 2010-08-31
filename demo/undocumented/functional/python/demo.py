"""This demo program computes the value of the functional

    M(v) = int v^2 + (grad v)^2 dx

on the unit square for v = sin(x) + cos(y). The exact
value of the functional is M(v) = 2 + 2*sin(1)*(1 - cos(1))

The functional M corresponds to the energy norm for a
simple reaction-diffusion equation."""

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-14 -- 2008-12-13"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008.

from dolfin import *

# Create mesh and define function space
mesh = UnitSquare(16, 16)
V = FunctionSpace(mesh, "CG", 2)

# Define the function v
v = Expression("sin(x[0]) + cos(x[1])", element=FiniteElement("CG", triangle, 2))

# Define functional
M = (v*v + dot(grad(v), grad(v)))*dx

# Evaluate functional
value = assemble(M, mesh=mesh)

exact_value = 2.0 + 2.0*sin(1.0)*(1.0 - cos(1.0))
print "The energy norm of v is: %.15g" % value
print "It should be:            %.15g" % exact_value
