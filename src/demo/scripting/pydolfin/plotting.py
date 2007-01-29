"Demonstrate plotting from within DOLFIN"

from dolfin import *

from numpy import zeros

# Example of a user-defined scalar function
class ScalarFunction(Function):

    def eval(self, p, i):
        return sin(5*p.x())*cos(3*p.x()*p.y())

f = ScalarFunction()

# Example of a user-defined vector function
class VectorFunction(Function):

    def __init__(self):
        Function.__init__(self, 2)
    
    def eval(self, p, i):
        if i == 0:
            return -p.y()
        else:
            return p.x()

g = VectorFunction()

# Attach a mesh to the functions
mesh = UnitSquare(16, 16)
f.attach(mesh)
g.attach(mesh)

# Plot mesh
plot(mesh)

# Plot scalar function
plot(f)

# Plot vector function
plot(g)
