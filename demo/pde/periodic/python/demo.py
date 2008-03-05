# This demo program solves Poisson's equation
#
#     - div grad u(x, y) = f(x, y)
#
# on the unit square with homogeneous Dirichlet boundary conditions
# at y = 0, 1 and periodic boundary conditions at x = 0, 1.
#
# Original implementation: ../cpp/main.cpp by Anders Logg
#
__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-15 -- 2007-11-28"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Create mesh and finite element
mesh = UnitSquare(32, 32)
element = FiniteElement("Lagrange", "triangle", 1)

# Source term
class Source(Function):
    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)

    def eval(self, values, x):
        dx = x[0] - 0.5
        dy = x[1] - 0.5
        values[0] = x[0]*sin(5.0*DOLFIN_PI*x[1]) + 1.0*exp(-(dx*dx + dy*dy)/0.02)

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool((x[1] < DOLFIN_EPS or x[1] > (1.0 - DOLFIN_EPS)) and on_boundary)

# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]


# Create Dirichlet boundary condition
u0 = Function(mesh, 0.0)
dirichlet_boundary = DirichletBoundary()
bc0 = DirichletBC(u0, mesh, dirichlet_boundary)

# Create periodic boundary condition
periodic_boundary = PeriodicBoundary()
bc1 = PeriodicBC(mesh, periodic_boundary)

# Collect boundary conditions
bcs = [bc0, bc1]

# Define variational problem
v = TestFunction(element)
u = TrialFunction(element)
f = Source(element, mesh)

a = dot(grad(v), grad(u))*dx
L = v*f*dx

# Solve PDE and plot solution
pde = LinearPDE(a, L, mesh, bcs)
u = pde.solve()
plot(u)

# Save solution to file
file = File("poisson.pvd")
file << u







