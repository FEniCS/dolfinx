"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)

and boundary conditions given by

    u(x, y) = 0        for x = 0 or x = 1
du/dn(x, y) = -sin(5*x) for y = 0 or y = 1

This demo is identical to the Poisson demo with the only difference that
the source and flux term is expressed using SpatialCoordinates in the
variational formulation.
"""

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@gmail.com)"
__date__ = "2010-03-05 -- 2010-03-05"
__copyright__ = "Copyright (C) 2010 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Create mesh and define function space
mesh = UnitSquare(32, 32)
V = FunctionSpace(mesh, "CG", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)
d_x = triangle.x[0] - 0.5
d_y = triangle.x[1] - 0.5
f = 10.0*exp(-(d_x*d_x + d_y*d_y) / 0.02)
g = -sin(5.0*triangle.x[0])
a = inner(grad(v), grad(u))*dx
L = v*f*dx + v*g*ds

# Compute solution
problem = VariationalProblem(a, L, bc)
u = problem.solve()

# Save solution in VTK format
file = File("spatial-coordinates.pvd")
file << u

# Plot solution
plot(u, interactive=True)
