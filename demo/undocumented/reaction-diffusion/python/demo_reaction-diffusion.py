"""This demo program solves the reaction-diffusion equation

    - div grad u + u = f

on the unit square with f = sin(x)*sin(y) and homogeneous Neumann
boundary conditions.

This demo is also available in more compact form in short.py,
the world's maybe shortest PDE solver.
"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2009-06-15 -- 2009-10-07"
__copyright__ = "Copyright (C) 2009 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Define variational problem
mesh = UnitSquare(32, 32)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("sin(x[0])*sin(x[1])")
a = dot(grad(u), grad(v))*dx + u*v*dx
L = f*v*dx

# Compute and plot solution
problem = VariationalProblem(a, L)
u = problem.solve()
plot(u, interactive=True)
