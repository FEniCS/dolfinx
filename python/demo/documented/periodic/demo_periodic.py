# Copyright (C) 2007 Kristian B. Oelgaard
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import matplotlib.pyplot as plt
from dolfin import *

# Source term
class Source(UserExpression):
    def eval(self, values, x):
        dx = x[0] - 0.5
        dy = x[1] - 0.5
        values[0] = x[0]*sin(5.0*DOLFIN_PI*x[1]) \
                    + 1.0*exp(-(dx*dx + dy*dy)/0.02)

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool((x[1] < DOLFIN_EPS or x[1] > (1.0 - DOLFIN_EPS)) \
                    and on_boundary)

# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]

# Create mesh and finite element
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicBoundary())

# Create Dirichlet boundary condition
u0 = Constant(0.0)
dbc = DirichletBoundary()
bc0 = DirichletBC(V, u0, dbc)

# Collect boundary conditions
bcs = [bc0]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Source(degree=1)
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bcs)

# Save solution to file
file = File("periodic.pvd")
file << u

# Plot solution
plot(u)
plt.show()
