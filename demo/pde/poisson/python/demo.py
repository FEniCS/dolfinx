"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 500*exp(-((x-0.5)^2 + (y-0.5)^2)/0.02)

and boundary conditions given by

    u(x, y)     = 0               for x = 0
    du/dn(x, y) = 25 cos(5 pi y)  for x = 1
    du/dn(x, y) = 0               otherwise
"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2007-08-16 -- 2008-12-05"
__copyright__ = "Copyright (C) 2007-2008 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Create mesh and FunctionSpace
mesh = UnitSquare(32, 32)
V = FunctionSpace(mesh, "Lagrange", 1)

# Neumann boundary condition, using python Functor
class Flux(Function):
    def eval(self, values, x):
        if x[0] > (1.0 - DOLFIN_EPS):
            values[0] = 25.0*cos(5.0*DOLFIN_PI*x[1])
        else:
            values[0] = 0.0

# Subdomain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < DOLFIN_EPS

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)
f = Function(V, cppexpr="500.0 * exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
g = Flux(V)
a = dot(grad(v), grad(u))*dx
L = v*f*dx + v*g*ds

# Define boundary condition
u0 = Constant("triangle", 0.0)
boundary = DirichletBoundary()
bc = DirichletBC(u0, V, boundary)

# Solve PDE and plot solution
pde = LinearPDE(a, L, bc, symmetric)
u = pde.solve()
plot(u, warpscalar=True, rescale=True)

# Save solution to file
file = File("poisson.pvd")
file << u

# Hold plot
interactive()
