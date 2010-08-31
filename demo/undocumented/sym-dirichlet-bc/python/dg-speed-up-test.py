# Efficiency test

__author__    = "Kent-Andre Mardal"
__date__      = "2008"
__copyright__ = "Copyright (C) 2008 Kent-Andre Mardal"
__license__   = "GNU LGPL Version 2.1"

import time
from dolfin import *

# Create mesh and finite element
mesh = UnitSquare(300,300)
V = FunctionSpace(mesh, "DG", 1)

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary)

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)
f = Constant(1)

# Normal component, mesh size and right-hand side
n = FacetNormal(mesh)
h = AvgMeshSize(mesh)

# Parameters
alpha = 4.0
gamma = 8.0

# Define boundary condition
u0 = Constant(0)
boundary = DirichletBoundary()
bc = DirichletBC(V, u0, boundary)

# Bilinear form
a = dot(grad(v), grad(u))*dx \
   - dot(avg(grad(v)), jump(u, n))*dS \
   - dot(jump(v, n), avg(grad(u)))*dS \
   + alpha/h('+')*dot(jump(v, n), jump(u, n))*dS \
   - dot(grad(v), mult(u, n))*ds \
   - dot(mult(v, n), grad(u))*ds \
   + gamma/h*v*u*ds

# Linear form
L = v*f*dx

backends = ["uBLAS", "PETSc", "Epetra"]

for backend in backends:
    if not has_la_backend(backend):
        print "DOLFIN not compiled with % linear algebra backend."%backend
        continue

    parameters["linear_algebra_backend"] = backend

    t0 = time.time()
    A, b = assemble_system(a, L, bc)
    t1 = time.time()
    print "time for new assembly      ", t1-t0, " with ", backend

    t0 = time.time()
    A = assemble(a)
    b = assemble(L)
    bc.apply(A, b)
    t1 = time.time()
    print "time for standard assembly ", t1-t0, " with ", backend



