"""This script demonstrates the L2 projection of a function onto a
non-matching mesh."""

__author__ = "Garth N. Wells (gnw20@cam.ac.uk)"
__date__ = "2009-10-10"
__copyright__ = "Copyright (C) 2009 Garth N. Wells"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

not_working_in_parallel("non-matching projection demo")

# Create mesh and define function spaces
mesh0 = UnitSquare(16, 16)
mesh1 = UnitSquare(64, 64)

# Create expression on P3
u0 = Expression("sin(10.0*x[0])*sin(10.0*x[1])", degree=3)

# Define projection space
P1 = FunctionSpace(mesh1, "CG", 1)

# Define projection variation problem
v  = TestFunction(P1)
u1 = TrialFunction(P1)
a  = v*u1*dx
L  = v*u0*dx

problem = VariationalProblem(a, L)
u1 = problem.solve()

# Plot functions
plot(u0, mesh=mesh0, title="u0")
plot(u1, title="u1")
interactive()

