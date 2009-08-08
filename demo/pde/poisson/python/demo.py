"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 500*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)

and boundary conditions given by

    u(x, y) = 0 for x = 0 or x = 1
"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2007-08-16 -- 2008-12-13"
__copyright__ = "Copyright (C) 2007-2008 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Create mesh and define function space
mesh = UnitSquare(32, 32)
V = FunctionSpace(mesh, "CG", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(mesh, 0.0)
bc = DirichletBC(V, u0, DirichletBoundary())

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)
f = Function(V, "500.0 * exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
a = inner(grad(v), grad(u))*dx
L = v*f*dx

# Compute solution
problem = VariationalProblem(a, L, bc)
u = problem.solve()

# FIXME: Temporary while testing parallel assembly
print "Norm of solution vector: %.15g" % u.vector().norm()

# Plot solution
plot(u)

# Save solution in VTK format
file = File("poisson.pvd")
file << u

# Hold plot
interactive()
