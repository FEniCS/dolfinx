"""This demo program solves Poisson's equation with a delta dirac point source

    - div grad u(x, y) = f(x, y)*dirac

on the unit square with source f given by

    f(x, y) = 0.4

and boundary conditions given by

    u(x, y) = 0          for x = 0 or x = 1
du/dn(x, y) = A*sin(5*x) for y = 0 or y = 1
"""

# Copyright (C) 2014 Johan Hake
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# Begin demo

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# Create mesh and define function space
mesh = UnitSquareMesh(MPI.comm_world, 128, 128)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return np.logical_or(x[:, 0] < DOLFIN_EPS, x[:, 0] > 1.0 - DOLFIN_EPS)

def center_func(x):
    r1 = np.logical_and(np.logical_and(0.45 <= x[:, 0], x[:, 0] <= 0.55), np.isclose(x[:, 1], 0.5))
    r2 = np.logical_and(np.logical_and(0.45 <= x[:, 1], x[:, 1] <= 0.55), np.isclose(x[:, 0], 0.5))
    return np.logical_or(r1, r2)

# Define domain for point integral
center_domain = MeshFunction("size_t", mesh, 0, 0)
center = AutoSubDomain(center_func)
center.mark(center_domain, 1)
dPP = dP(subdomain_data=center_domain)

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = 0.4
x = SpatialCoordinate(mesh)
g = 10.0 * sin(5 * x[0])
a = inner(grad(u), grad(v))*dx
L = f*v*dPP(1) + g*v*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution in VTK format
file = File("poisson.pvd")
file << u

# Plot solution
plot(u)
plt.show()
