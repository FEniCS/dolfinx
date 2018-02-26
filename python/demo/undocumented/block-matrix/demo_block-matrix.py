"""This demo illustrates basic usage of block matrices and vectors."""

# Copyright (C) 2008 Kent-Andre Mardal
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


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
