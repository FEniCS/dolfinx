"""This script demonstrates how to interpolate functions between different
finite element spaces on non-matching meshes."""

__author__ = "Garth N. Wells (gnw20@cam.ac.uk)"
__date__ = "2009-05-19"
__copyright__ = "Copyright (C) 2009 Garth N. Wells"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

not_working_in_parallel("non-matching interpolation demo")

# Create mesh and define function spaces
mesh0 = UnitSquare(16, 16)
mesh1 = UnitSquare(64, 64)

P1 = FunctionSpace(mesh1, "CG", 1)
P3 = FunctionSpace(mesh0, "CG", 3)

# Define function
v0 = Expression("sin(10.0*x[0])*sin(10.0*x[1])", element=FiniteElement('CG', triangle, 3))
v1 = Function(P1)

# Interpolate
v1.interpolate(v0)

# Plot functions
plot(v0, mesh=mesh0, title="v0")
plot(v1, title="v1")
interactive()

print norm(v0, mesh = mesh1)
print norm(v1)
