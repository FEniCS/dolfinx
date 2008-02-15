# This program illustrates the use of the DOLFIN for solving a nonlinear PDE
# by solving the nonlinear variant of Poisson's equation
#
#     - div (1+u^2) grad u(x, y) = f(x, y)
#
# on the unit square with source f given by
#
#     f(x, y) = t * x * sin(y)
#
# and boundary conditions given by
#
#     u(x, y)     = t  for x = 0
#     du/dn(x, y) = 0  otherwise
#
# where t is pseudo time.
#
# This is equivalent to solving: 
# F(u) = (grad(v), (1-u^2)*grad(u)) - f(x,y) = 0
#
# Original implementation: ../cpp/main.cpp by Garth N. Wells
#

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-14 -- 2007-11-28"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *


# FIXME: Not working, see notice below
import sys
print "This demo is not working, please fix me"
sys.exit(1)

# Create mesh and finite element
mesh = UnitSquare(16, 16)
element = FiniteElement("Lagrange", "triangle", 1)

# Source term
class Source(Function, TimeDependent):
    def __init__(self, element, mesh, t):
        Function.__init__(self, element, mesh)
        TimeDependent.__init__(self, t)

    def eval(self, values, x):
        values[0] = time()*x[0]*sin(x[1])

# Dirichlet boundary condition
class DirichletBoundaryCondition(Function, TimeDependent):
    def __init__(self, element, mesh, t):
        Function.__init__(self, element, mesh)
        TimeDependent.__init__(self, t)

    def eval(self, values, x):
      values[0] = 1.0*time()

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary)


# Pseudo time
t = 0.0

# Create source function
f = Source(element, mesh, t)

# Dirichlet boundary conditions
dirichlet_boundary =  DirichletBoundary()
g = DirichletBoundaryCondition(element, mesh, t)
bc = DirichletBC(g, mesh, dirichlet_boundary)

# Solution function
u = Function(element, mesh)


v = TestFunction(element)
U = TrialFunction(element)
U0= Function(element, mesh)

a = v.dx(i)*(1.0 + U0*U0)*U.dx(i)*dx + v.dx(i)*2.0*U0*U*U0.dx(i)*dx
L = v*f*dx - v.dx(i)*(1.0 + U0*U0)*U0.dx(i)*dx

pde = NonlinearPDE(a, L, mesh, bc)

# ERROR:
# Traceback (most recent call last):
#   File "demo.py", line 80, in <module>
#     pde = NonlinearPDE(a, L, mesh, bc)
# NameError: name 'NonlinearPDE' is not defined

# We need to define NonlinearPDE in assemble.py.
# However, we need to solve the issues with NonlinearProblem first.

# # Solve nonlinear problem in a series of steps
# dt = 1.0
# T  = 3.0

# #  pde.set("Newton relative tolerance", 1e-6); 
# #  pde.set("Newton convergence criterion", "incremental"); 

# # Solve
# u =  pde.solve(U0, t, T, dt);

# # Plot solution
# plot(u)

# # Save function to file
# file = File("nonlinear_poisson.pvd")
# file << u


