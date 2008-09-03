"""This demo demonstrate speed-up for the standard Poisson problem
   (without Python callbacks)
"""

__author__ = "Kent-Andre Mardal (kent-and@simula.no)"
__date__ = "2008-08-13"
__copyright__ = "Copyright (C) 2008 Kent-Andre Mardal"
__license__  = "GNU LGPL Version 2.1"

import time
from dolfin import *

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

mesh = UnitSquare(500,500)
element = FiniteElement("Lagrange", "triangle", 1)


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


backends = ["uBLAS", "PETSc", "Epetra"]

for backend in backends: 
    dolfin_set("linear algebra backend", backend)

    t0 = time.time()
    A = assemble(a, mesh)
    b = assemble(L, mesh)
    bc.apply(A, b, a)
    t1 = time.time()
    print "time for standard assembly ", t1-t0, " using ", backend

    t0 = time.time()
    A, b = assemble_system(a, L, bc, mesh)
    t1 = time.time()
    print "time for new assembly      ", t1-t0, " using ", backend


#summary()
