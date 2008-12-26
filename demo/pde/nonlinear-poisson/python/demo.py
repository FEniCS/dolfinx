"""This demo illustrates how to use of DOLFIN for solving a nonlinear
PDE, in this case a nonlinear variant of Poisson's equation,

    - div (1 + u^2) grad u(x, y) = f(x, y)

on the unit square with source f given by

     f(x, y) = x*sin(y)

and boundary conditions given by

     u(x, y)     = 1  for x = 0
     du/dn(x, y) = 0  otherwise

This is equivalent to solving

    F(u) = (grad(v), (1 + u^2)*grad(u)) - (v, f) = 0

"""

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-14 -- 2008-12-26"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

# Original implementation (../cpp/main.cpp) by Garth N. Wells.
# Modified by Anders Logg, 2008.

from dolfin import *

# Create mesh and define function space
mesh = UnitSquare(16, 16)
V = FunctionSpace(mesh, "CG", 1)

# Dirichlet boundary condition
class DirichletBoundaryCondition(Function, TimeDependent):
    def __init__(self, V, t):
        Function.__init__(self, V)
        TimeDependent.__init__(self, t)

    def eval(self, values, x):
      values[0] = 1.0*time()

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary

# Pseudo-time
t = 0.0

# Create source function
f = Function(V, "t*x[0]*sin(x[1])")

# Dirichlet boundary conditions
g = Function(V, "t")
bc = DirichletBC(V, g, DirichletBoundary())

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)
U = Function(V)
a = v.dx(i)*(1.0 + U*U)*u.dx(i)*dx + v.dx(i)*(2.0*U*u)*U.dx(i)*dx
L = v.dx(i)*(1.0 + U*U)*U.dx(i)*dx - v*f*dx
#L = v*f*dx - v.dx(i)*(1.0 + U*U)*u.dx(i)*dx

pde = NonlinearPDE(a, L, mesh, bc)

# Solve nonlinear problem in a series of steps
dt = 1.0
T  = 3.0
pde.set("Newton relative tolerance", 1e-6)
pde.set("Newton convergence criterion", "incremental")
u = pde.solve(U0, t, T, dt)

# Plot solution
plot(u)

# Save function to file
file = File("nonlinear_poisson.pvd")
file << u
