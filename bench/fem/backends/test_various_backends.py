__author__ = "Kent-Andre Mardal (kent-and@simula.no)"
__date__ = "2008-04-30"
__copyright__ = "Copyright (C) 2007 Kent-Andre Mardal"
__license__  = "GNU LGPL Version 2.1"



from dolfin import *
from time import time

N = 1000
mesh = UnitSquare(N,N)
V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)
u = TrialFunction(V)

a = dot(v, u)*dx
t0 = time()
backend = PETScFactory.instance()
A = assemble(a, backend=backend)
t1 = time()
print "time (PETSc)", t1-t0


a = dot(v, u)*dx
t0 = time()
backend = EpetraFactory.instance()
A = assemble(a, backend=backend)
t1 = time()
print "time (Epetra) ", t1-t0

a = dot(v, u)*dx
t0 = time()
backend = uBLASSparseFactory.instance()
A = assemble(a, backend=backend)
t1 = time()
print "time (uBLAS) ", t1-t0

a = dot(v, u)*dx
t0 = time()
backend = MTL4Factory.instance()
A = assemble(a, backend=backend)
t1 = time()
print "time (MTL4) ", t1-t0








