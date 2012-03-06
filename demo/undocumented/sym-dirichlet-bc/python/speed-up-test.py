"""This demo demonstrate speed-up for the standard Poisson problem
   (without Python callbacks)
"""

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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2008-08-13
# Last changed: 2008-08-13

from dolfin import *
import time


# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary and x[0] < DOLFIN_EPS)

mesh = UnitCube(32,32,32)
V = FunctionSpace(mesh, "CG", 1)

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)

f = Expression("500.0*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")


a = dot(grad(v), grad(u))*dx
L = v*f*dx

# Define boundary condition
u0 = Constant(0)
boundary = DirichletBoundary()
bc = DirichletBC(V, u0, boundary)


backends = ["uBLAS", "PETSc", "Epetra"]

for backend in backends:
    if not has_linear_algebra_backend(backend):
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

    t0 = time.time()
    A, Aa = symmetric_assemble(a, bcs=bc)
    b = assemble(L, bcs=bc)
    t1 = time.time()
    print "time for symm assembly     ", t1-t0, " using ", backend

#summary()
