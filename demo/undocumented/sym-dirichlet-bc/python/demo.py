"""This demo demonstrate how to assemble a linear system including
boundary conditions."""

# Modified by Kristian Oelgaard, 2008

__author__ = "Kent-Andre Mardal (kent-and@simula.no)"
__date__ = "2008-08-13 -- 2009-10-07"
__copyright__ = "Copyright (C) 2008 Kent-Andre Mardal"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Create mesh and finite element
mesh = UnitSquare(32, 32)
V = FunctionSpace(mesh, "CG", 1)

# Source term
class Source(Expression):
    def eval(self, values, x):
        dx = x[0] - 0.5
        dy = x[1] - 0.5
        values[0] = 500.0*exp(-(dx*dx + dy*dy)/0.02)

# Neumann boundary condition
class Flux(Expression):
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
v = TestFunction(V)
u = TrialFunction(V)
f = Source(V)
g = Flux(V)

a = inner(grad(v), grad(u))*dx
L = v*f*dx + v*g*ds

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, DirichletBoundary())

# Solve PDE and plot solution
problem = VariationalProblem(a, L, bc)
U = problem.solve()

plot(U)

# Save solution to file
file = File("poisson.pvd")
file << U

# Hold plot
interactive()

summary()
