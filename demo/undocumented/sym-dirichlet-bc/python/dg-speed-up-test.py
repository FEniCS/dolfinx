# Efficiency test

# Copyright (C) 2008 Kent-Andre Mardal
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2008
# Last changed: 2008

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



