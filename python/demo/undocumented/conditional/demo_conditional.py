"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) =    -1.0 if (x - 0.33)^2 + (y - 0.67)^2 < 0.015
                  5.0 if 0.015 < (x - 0.33)^2 + (y - 0.67)^2 < 0.025
                 -1.0 if (x,y) in triangle( (0.55, 0.05), (0.95, 0.45), (0.55, 0.45) )
                  0.0 otherwise

and homogeneous Dirichlet boundary conditions.
"""

# Copyright (C) 2010 Kristian B. Oelgaard
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *
import matplotlib.pyplot as plt


# Create mesh and define function space
mesh = UnitSquareMesh(64, 64)
V = FunctionSpace(mesh, "CG", 2)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
x = SpatialCoordinate(mesh)

c0 = conditional(le( (x[0]-0.33)**2 + (x[1]-0.67)**2,  0.015), -1.0, 5.0)
c = conditional( le( (x[0]-0.33)**2 + (x[1]-0.67)**2,  0.025), c0, 0.0 )

t0 = conditional(ge( x[0],  0.55), -1.0, 0.0)
t1 = conditional(le( x[0],  0.95), t0, 0.0)
t2 = conditional(ge( x[1],  0.05), t1, 0.0)
t3 = conditional(le( x[1],  0.45), t2, 0.0)
t = conditional(ge( x[1] - x[0] - 0.05 + 0.55,  0.0), t3, 0.0)
f = c + t
a = inner(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution in VTK format
file = File("conditional.pvd")
file << u

# Plot solution
plot(u)
plt.show()
