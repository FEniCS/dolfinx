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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2008.
#
# First added:  2008-12-12
# Last changed: 2008-12-12

from dolfin import *

# Create a simple stiffness matrix
mesh = UnitSquare(4, 4)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v))*dx
A = assemble(a)

# Create a block matrix
AA = BlockMatrix(2, 2)
AA[0, 0] = A
AA[1, 0] = A
AA[0, 1] = A
AA[1, 1] = A

# Create a block vector
x = Vector(A.size(0))
for i in range(x.size()):
    x[i] = i
xx = BlockVector(2)
xx[0] = x
xx[1] = x

# Create another block vector
y = Vector(A.size(1))
yy = BlockVector(2)
yy[0] = y
yy[1] = y

# Multiply
AA.mult(xx, yy)
print "||Ax|| =", y.norm("l2")
