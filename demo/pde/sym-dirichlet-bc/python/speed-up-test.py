"""This demo demonstrate speed-up for the standard Poisson problem
   (without Python callbacks)
"""

__author__ = "Kent-Andre Mardal (kent-and@simula.no)"
__date__ = "2008-08-13"
__copyright__ = "Copyright (C) 2008 Kent-Andre Mardal"
__license__  = "GNU LGPL Version 2.1"

import time
from dolfin import *


# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary and x[0] < DOLFIN_EPS)

mesh = UnitSquare(500,500)
V = FunctionSpace(mesh, "CG", 1)

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)

f = Function(V, "500.0*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")


a = dot(grad(v), grad(u))*dx
L = v*f*dx

# Define boundary condition
u0 = Constant(0)
boundary = DirichletBoundary()
bc = DirichletBC(V, u0, boundary)


backends = ["uBLAS", "PETSc", "Epetra"]

for backend in backends:
    if not has_la_backend(backend):
        print "DOLFIN not compiled with % linear algebra backend."%backend
        continue

    parameters["linear_algebra_backend"] = backend

    t0 = time.time()
    A = assemble(a)
    b = assemble(L)
    bc.apply(A, b)
    t1 = time.time()
    print "time for standard assembly ", t1-t0, " using ", backend

    t0 = time.time()
    A, b = assemble_system(a, L, bc)
    t1 = time.time()
    print "time for new assembly      ", t1-t0, " using ", backend


#summary()
