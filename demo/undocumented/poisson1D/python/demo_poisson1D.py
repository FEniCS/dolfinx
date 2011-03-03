"""This demo program solves Poisson's equation

    - div grad u(x) = f(x)

on the unit interval with source f given by

    f(x) = 9*pi^2*sin(3*pi*x[0])

and boundary conditions given by

    u(x) = 0 for x = 0
    du/dx = 0 for x = 1
"""

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-28 -- 2009-10-07"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Create mesh and function space
mesh = UnitInterval(50)
V = FunctionSpace(mesh, "CG", 1)

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] < DOLFIN_EPS

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)
f = Expression("9.0*pi*pi*sin(3.0*pi*x[0])")
g = Expression("3.0*pi*cos(3.0*pi*x[0])")

a = dot(grad(v), grad(u))*dx
L = v*f*dx + v*g*ds

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, DirichletBoundary())

# Solve PDE and plot solution
problem = VariationalProblem(a, L, bc)
u = problem.solve()

# Save solution to file
file = File("poisson.pvd")
file << u

# Plot solution
plot(u, interactive=True)
