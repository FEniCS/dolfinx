__author__ = "Kent-Andre Mardal (kent-and@simula.no)"
__date__ = "2008-04-30"
__copyright__ = "Copyright (C) 2007 Kent-Andre Mardal"
__license__  = "GNU LGPL Version 2.1"



from dolfin import *
from time import time

N = 1000 
mesh = UnitSquare(N,N) 
element = FiniteElement("CG", "triangle", 1)
v = TestFunction(element)
u = TrialFunction(element)

a = dot(v, u)*dx 
t0 = time()
backend = PETScFactory.instance()
A = assemble(a, mesh, backend=backend)
t1 = time()
print "time ", t1-t0


a = dot(v, u)*dx 
t0 = time()
backend = EpetraFactory.instance()
A = assemble(a, mesh, backend=backend)
t1 = time()
print "time ", t1-t0

a = dot(v, u)*dx 
t0 = time()
backend = uBlasFactory.instance()
A = assemble(a, mesh, backend=backend)
t1 = time()
print "time ", t1-t0









