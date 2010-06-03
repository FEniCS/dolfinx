"""
This demo program illustrates how to solve Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with pure Neumann boundary conditions:

    du/dn(x, y) = -sin(5*x)

and source f given by

    f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)

Since only Neumann conditions are applied, u is only determined up to
a constant c by the above equations. An addition constraint is thus
required, for instance

  \int u = 0

This can be accomplished by introducing the constant c as an
additional unknown (to be sought in the space of real numbers)
and the above constraint.
"""

__author__ = "Marie E. Rognes (meg@simula.no)"
__date__ = "2010-05-10"
__copyright__ = "Copyright (C) 2010 Marie E. Rognes"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

not_working_in_parallel("neumann-poisson demo (with space of reals)")

# Create mesh and define function space
mesh = UnitSquare(64, 64)
V = FunctionSpace(mesh, "CG", 1)
R = FunctionSpace(mesh, "R", 0)
W = V * R

# Define variational problem
(v, d) = TestFunctions(W)
(u, c) = TrialFunction(W)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
g = Expression("-sin(5*x[0])")
a = (inner(grad(v), grad(u)) + v*c + d*u)*dx
L = v*f*dx + v*g*ds

# Compute solution
problem = VariationalProblem(a, L)
(u, c) = problem.solve()

# Plot solution
plot(u, interactive=True)
