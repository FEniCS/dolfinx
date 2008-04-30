
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









