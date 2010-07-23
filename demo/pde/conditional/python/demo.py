"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) =    -1.0 if (x - 0.33)^2 + (y - 0.67)^2 < 0.015
                  5.0 if 0.015 < (x - 0.33)^2 + (y - 0.67)^2 < 0.025
                  0.0 otherwise

and homogeneous Dirichlet boundary conditions.
"""

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@gmail.com)"
__date__ = "2010-07-23 -- 2010-07-23"
__copyright__ = "Copyright (C) 2010 Kristian B. Oelgaard"
__license__  = "GNU GPL Version 3.0 or later"

from dolfin import *

# Create mesh and define function space
mesh = UnitSquare(64, 64)
V = FunctionSpace(mesh, "CG", 2)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)
x = V.cell().x
c0 = conditional(le( (x[0]-0.33)**2 + (x[1]-0.67)**2,  0.015), -1.0, 5.0)
f = conditional( le( (x[0]-0.33)**2 + (x[1]-0.67)**2,  0.025), c0, 0.0 )
a = inner(grad(v), grad(u))*dx
L = v*f*dx

# Compute solution
problem = VariationalProblem(a, L, bc)
u = problem.solve()

# Save solution in VTK format
file = File("conditional.pvd")
file << u

# Plot solution
plot(u, interactive=True)
