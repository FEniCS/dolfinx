"""This demo illustrates basic usage of block matrices and vectors."""

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
# Modified by Anders Logg, 2008.
#
# First added:  2008-12-12
# Last changed: 2012-11-12


from dolfin import *

# Create a simple stiffness matrix and vector
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v))*dx
A = assemble(a)
x0 = SpatialCoordinate(mesh)[0]
L = sin(x0)*v*dx
x = assemble(L)

# Create a block matrix
AA = BlockMatrix(2, 2)
AA[0, 0] = A
AA[1, 0] = A
AA[0, 1] = A
AA[1, 1] = A

# Create a block vector (that is compatible with A in parallel)
xx = BlockVector(2)
xx[0] = x
xx[1] = x

# Create a another block vector (that is compatible with A in parallel)
y = Vector()
A.init_vector(y, 0)
yy = BlockVector(2)
yy[0] = y
yy[1] = y

# Multiply
AA.mult(xx, yy)
print("||Ax|| =", y.norm("l2"))
