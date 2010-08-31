"""This demo illustrates basic usage of block matrices and vectors."""

__author__ = "Kent-Andre Mardal (kent-and@simula.no)"
__date__ = "2008-12-12 -- 2008-12-12"
__copyright__ = "Copyright (C) 2008 Kent-Andre Mardal"
__license__  = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008.

from dolfin import *

# Create a simple stiffness matrix
mesh = UnitSquare(4, 4)
V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)
u = TrialFunction(V)
a = dot(grad(v), grad(u))*dx
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
