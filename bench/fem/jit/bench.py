"""This script provides a benchmark for the JIT compiler, in
particular the speed of the in-memory cache."""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2008-09-04 -- 2008-09-04"
__copyright__ = "Copyright (C) 2008 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *
from time import time

mesh = UnitCube(3, 3, 3)
shape = "tetrahedron"

h = MeshSize(shape, mesh)
dt = 0.1

V = VectorElement("Lagrange", shape, 1)
Q = FiniteElement("Lagrange", shape, 1)
DG = FiniteElement("DG", shape, 0)
DGv = VectorElement("DG", shape, 0)

v = TestFunction(V)
q = TestFunction(Q)
z = TestFunction(DGv)
u = TrialFunction(V)
p = TrialFunction(Q)
w = TrialFunction(DGv)

u0 = Function(V, mesh, Vector())
u1 = Function(V, mesh, Vector())
p1 = Function(Q, mesh, Vector())
W  = Function(DGv, mesh, Vector())
nu = Function(DG, mesh, 0.1)
k  = Function(DG, mesh, dt)
d1 = h
d2 = 2.0*h

U = 0.5*(u0 + u)
F = (1.0/k)*dot(v, u - u0) + dot(v, mult(grad(U), W)) + nu*dot(grad(v), grad(U)) - div(v)*p1 + \
    d1*dot(mult(grad(v), W), mult(grad(U), W)) + d2*div(v)*div(U)
a = lhs(F*dx)

# Assemble once
A = assemble(a, mesh)

# Then assemble some more
t = time()
for i in range(100):
    A = assemble(a, mesh)
t = time() - t

print "Elapsed time: " + str(t)
