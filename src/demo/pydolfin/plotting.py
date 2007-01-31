"Demonstrate plotting from within DOLFIN"

from dolfin import *

# Example of a user-defined scalar function
class ScalarFunction(Function):

    def eval(self, x, i):
        return x[0]*x[1] + 1.0 / (x[0] + 0.1)

f = ScalarFunction()

# Example of a user-defined vector function
class VectorFunction2D(Function):

    def __init__(self):
        Function.__init__(self, 2)
    
    def eval(self, x, i):
        if i == 0:
            return -x[1]**2
        else:
            return x[0]

g = VectorFunction2D()

# Example of a user-defined vector function
class VectorFunction3D(Function):
    
    def __init__(self):
        Function.__init__(self, 3)

    def eval(self, x, i):
        if i == 0:
            return -x[1]**2
        elif i == 1:
            return x[2]**2
        else:
            return x[0]**2

h = VectorFunction3D()

# Attach mesh for 2D plots
mesh = UnitSquare(16, 16)
f.attach(mesh)
g.attach(mesh)

# Plot mesh
plot(mesh)

# Plot scalar function
plot(f)

# Plot vector function
plot(g)

# Plot mesh displaced by vector function
plot(mesh, g)

# Attach mesh for 3D plots
mesh = UnitCube(16,16,16)
f.attach(mesh)
h.attach(mesh)

# Plot mesh
plot(mesh)

# Plot scalar function
plot(f)

# Plot vector function
plot(h)

# Plot mesh displaced by vector function
plot(mesh, h)
