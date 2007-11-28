# This demo program solves Poisson's equation
#
#     - div grad u(x) = f(x)
#
# on the unit square with source f given by
#
#     f(x) = 9.0*DOLFIN_PI * DOLFIN_PI * sin(3.0*DOLFIN_PI*x[0]);
#
# and boundary conditions given by
#
#     u(x)     = 0  for x = 0 and x = 1

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-28 -- 2007-11-28"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Create mesh and finite element
mesh = UnitInterval(15)

element = FiniteElement("Lagrange", "interval", 1)

# Source term
class Source(Function):
    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)
    def eval(self, values, x):
        values[0] = 9.0*DOLFIN_PI * DOLFIN_PI * sin(3.0*DOLFIN_PI*x[0])

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary)

# Define variational problem
v = TestFunction(element)
u = TrialFunction(element)
f = Source(element, mesh)

a = dot(grad(v), grad(u))*dx
L = v*f*dx

# Define boundary condition
u0 = Function(mesh, 0.0)
boundary = DirichletBoundary()
bc = DirichletBC(u0, mesh, boundary)

# Solve PDE and plot solution
pde = LinearPDE(a, L, mesh, bc)
u = pde.solve()

plot(u)

# Save solution to file
file = File("poisson.pvd")
file << u

