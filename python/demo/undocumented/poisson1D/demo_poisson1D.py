"""This demo program solves Poisson's equation

    - div grad u(x) = f(x)

on the unit interval with source f given by

    f(x) = 9*pi^2*sin(3*pi*x[0])

and boundary conditions given by

    u(x) = 0 for x = 0
    du/dx = 0 for x = 1
"""

# Copyright (C) 2007 Kristian B. Oelgaard
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *
import matplotlib.pyplot as plt


# Create mesh and function space
mesh = UnitIntervalMesh(50)
V = FunctionSpace(mesh, "CG", 1)

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] < DOLFIN_EPS

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("9.0*pi*pi*sin(3.0*pi*x[0])", degree=2)
g = Expression("3.0*pi*cos(3.0*pi*x[0])", degree=2)

a = dot(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, DirichletBoundary())

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution to file
file = File("poisson.pvd")
file << u

# Plot solution
plot(u)
plt.show()
