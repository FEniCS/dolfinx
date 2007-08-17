# This demo program solves Poisson's equation
#
#     - div grad u(x, y) = f(x, y)
#
# on the unit square with source f given by
#
#     f(x, y) = 500*exp(-((x-0.5)^2 + (y-0.5)^2)/0.02)
#
# and boundary conditions given by
#
#     u(x, y)     = 0  for x = 0
#     du/dn(x, y) = 1  for x = 1
#     du/dn(x, y) = 0  otherwise

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2007-08-16 -- 2007-08-18"
__copyright__ = "Copyright (C) 2007 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

print "EXPERIMENTAL: This demo does not work yet"

# Things to fix:
# - Typemaps for double* and const double*
# - Memory leak warning for Array of Function*
# - Simpler definition of functions without needing to create classes

# Create mesh and finite element
mesh = UnitSquare(32, 32)
element = FiniteElement("Lagrange", "triangle", 1)

# Source term
class Source(Function):

    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)

    def eval(self, x):
        return 1.0
        # FIXME: Typemap error for x
        #dx = x[0] - 0.5
        #dy = x[1] - 0.5
        #return 500.0*exp(-(dx*dx + dy*dy)/0.02)

# Neumann boundary condition
class Flux(Function):

    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)

    def eval(self, x):
        return 1.0
        # FIXME: Typemap error for x
        #if x[0] > DOLFIN_EPS:
        #    return 25.0*sin(5.0*DOLFIN_PI*x[1])
        #else:
        #    return 0.0

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return True
        # FIXME: Typemap error for x
        # return x[0] < DOLFIN_EPS and on_boundary

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
u0 = cpp_Function(mesh, 0.0)
boundary = DirichletBoundary()
bc = DirichletBC(u0, mesh, boundary)

# Solve PDE and plot solution
pde = LinearPDE(a, L, mesh, bc)
u = pde.solve()
plot(u)

# Save solution to file
file = File("poisson.pvd")
file << u

print "EXPERIMENTAL: This demo does not work yet"
