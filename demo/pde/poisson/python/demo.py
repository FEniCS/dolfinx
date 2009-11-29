"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)

and boundary conditions given by

    u(x, y) = 0        for x = 0 or x = 1
du/dn(x, y) = sin(5*x) for y = 0 or y = 1
"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2007-08-16 -- 2009-11-24"
__copyright__ = "Copyright (C) 2007-2009 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Create mesh and define function space
mesh = UnitSquare(32, 32)
V = FunctionSpace(mesh, "CG", 1)

class Source(Expression):
    def eval(self, values, x):
        values[0] = 4.0*DOLFIN_PI*DOLFIN_PI*DOLFIN_PI*DOLFIN_PI*sin(DOLFIN_PI*x[0])*sin(DOLFIN_PI*x[1])

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)
#f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
f = Source(V)
g = Expression("sin(5*x[0])")
a = inner(grad(v), grad(u))*dx
L = v*f*dx - v*g*ds

# Compute solution
problem = VariationalProblem(a, L, bc)
u = problem.solve()

# Save solution in VTK format
file = File("poisson.pvd")
file << u

# Plot solution
#plot(u, interactive=True)
