"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 500*exp(-((x-0.5)^2 + (y-0.5)^2)/0.02)

and boundary conditions given by

    u(x, y)     = 0               for x = 0
    du/dn(x, y) = 25 sin(5 pi y)  for x = 1
    du/dn(x, y) = 0               otherwise
"""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2007-08-16 -- 2008-04-03"
__copyright__ = "Copyright (C) 2007-2008 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Create mesh and finite element
mesh = UnitSquare(500,500)
element = FiniteElement("Lagrange", "triangle", 1)

# Source term
class Source(Function):
    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)
    def eval(self, values, x):
        dx = x[0] - 0.5
        dy = x[1] - 0.5
        values[0] = 500.0*exp(-(dx*dx + dy*dy)/0.02)

# Neumann boundary condition
class Flux(Function):
    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)
    def eval(self, values, x):
        if x[0] > DOLFIN_EPS:
            values[0] = 25.0*sin(5.0*DOLFIN_PI*x[1])
        else:
            values[0] = 0.0


# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary and x[0] < DOLFIN_EPS)

# Define variational problem
v = TestFunction(element)
u = TrialFunction(element)
#f = Source(element, mesh)
#g = Flux(element, mesh)
f = Function(element, mesh, 1.0)
g = Function(element, mesh, 1.0)



a = dot(grad(v), grad(u))*dx
L = v*f*dx + v*g*ds

# Define boundary condition
u0 = Function(mesh, 0.0)
boundary = DirichletBoundary()
bc = DirichletBC(u0, mesh, boundary)

# Solve PDE and plot solution

import time

t0 = time.time()
A = assemble(a, mesh)
b = assemble(L, mesh)
bc.apply(A, b, a)
t1 = time.time()
print "time for standard assembly ", t1-t0



t0 = time.time()
A, b = assemble_system(a, L, bc, mesh)
t1 = time.time()
print "time for new assembly ", t1-t0

#summary()
