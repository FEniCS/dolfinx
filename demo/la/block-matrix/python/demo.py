
from dolfin import *

mesh = UnitSquare(12,12)

element = FiniteElement("Lagrange", "triangle", 1)
v = TestFunction(element)
u = TrialFunction(element)
a = dot(grad(v), grad(u))*dx
A = assemble(a, mesh)

AA = BlockMatrix(2,2)
AA[0,0] = A
AA[1,0] = A
AA[0,1] = A
AA[1,1] = A

x = Vector(A.size(0))
xx = BlockVector(2) 
xx[0] = x 
xx[1] = x 

y = Vector(A.size(1))
yy = BlockVector(2) 
yy[0] = y 
yy[1] = y 

AA.mult(xx,yy)


