""" This demo implements a Poisson equations solver
based on the demo "dolfin/demo/pde/poisson/python/demo.py"
in Dolfin using Epetra matrices, the AztecOO CG solver and ML
AMG preconditioner
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
# First added:  2008-04-24
# Last changed: 2008-04-24

from dolfin import *

if not has_la_backend("Epetra"):
    print "*** Warning: Dolfin is not compiled with Trilinos linear algebra backend"
    print "Exiting."
    exit()

parameters["linear_algebra_backend"] = "Epetra"

# Create mesh and finite element
mesh = UnitSquare(20,20)
V = FunctionSpace(mesh, "Lagrange", 1)


# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary and x[0] < DOLFIN_EPS)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("500.0 * exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")

a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Define boundary condition
u0 = Constant(0)
bc = DirichletBC(V, u0, DirichletBoundary())

# Create linear system
A, b = assemble_system(a, L, bc)

# Create solution vector (also used as start vector)
U = Function(V)

solve(A, U.vector(), b, "cg", "ilu")

# plot the solution
plot(U)
interactive()

# Save solution to file
file = File("poisson.pvd")
file << U



