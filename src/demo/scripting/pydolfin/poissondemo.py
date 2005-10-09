from dolfin import *
from math import *

class Source(Function):
    def __call__(self, point):
        print "Evaluating Python source function: "
        print "x: " + str(point.x),
        print "y: " + str(point.y),
        print "z: " + str(point.z)
        return pi * pi * sin(pi * point.x)

class SimpleBC(BoundaryCondition):
    def __call__(self, point):
        print "Evaluating Python BC: "
        print "x: " + str(point.x),
        print "y: " + str(point.y),
        print "z: " + str(point.z)
        
        value = BoundaryValue()
        if point.x == 0.0 or point.x == 1.0:
            value.set(0.0)
        return value
    
f = Source()
bc = SimpleBC()
mesh = UnitSquare(30, 30)

PoissonSolver_solve(mesh, f, bc)

meshfile = File("square01.xml.gz")
meshfile << mesh

