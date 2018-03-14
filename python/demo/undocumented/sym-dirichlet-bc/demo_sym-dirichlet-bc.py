"""This demo demonstrate how to assemble a linear system including
boundary conditions."""

# Copyright (C) 2008 Kent-Andre Mardal
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *
import matplotlib.pyplot as plt

# Create mesh and finite element
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "CG", 1)

# Source term
class Source(UserExpression):
    def eval(self, values, x):
        dx = x[0] - 0.5
        dy = x[1] - 0.5
        values[0] = 500.0*exp(-(dx*dx + dy*dy)/0.02)

# Neumann boundary condition
class Flux(UserExpression):
    def eval(self, values, x):
        if x[0] > DOLFIN_EPS:
            values[0] = 25.0*sin(5.0*DOLFIN_PI*x[1])
        else:
            values[0] = 0.0

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] < DOLFIN_EPS

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Source(degree=2)
g = Flux(degree=2)
a = inner(grad(u), grad(v))*dx
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
