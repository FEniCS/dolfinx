from dolfin import *
from math import *

class Source(Function):
    def __call__(self, point):
#        return pi * pi * sin(pi * point.x)
        return point.y

class SimpleBC(BoundaryCondition):
    def __call__(self, point):
        value = BoundaryValue()
        if point.x == 0.0 or point.x == 1.0:
            value.set(0.0)
        return value
    
f = Source()
bc = SimpleBC()
mesh = UnitSquare(2, 2)

a = PoissonBilinearForm()
L = PoissonLinearForm(f)

A = Matrix()
x = Vector()
b = Vector()

FEM_assemble(a, L, A, b, mesh, bc)

print "A:"
A.disp(False)

print "b:"
b.disp()

# Linear algebra could in certain cases be handled by Python modules,
# Numeric for example.

linearsolver = GMRES()
linearsolver.solve(A, x, b)

print "x:"
x.disp()

trialelement = PoissonBilinearFormTrialElement()
u = Function(x, mesh, trialelement)
file = File("poisson.m")
file << u

# Plotting should also be handled by Python modules

vtkfile = File("poisson.vtk", File.VTK)
vtkfile << u
