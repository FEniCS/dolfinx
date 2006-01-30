from dolfin import *
from math import *

class Source(Function):
    def eval(self, point, i):
        return point.y + 1.0

class SimpleBC(BoundaryCondition):
    def eval(self, value, point, i):
        if point.x == 0.0 or point.x == 1.0:
            value.set(0.0)
        return value
    
f = Source()
bc = SimpleBC()
mesh = UnitSquare(10, 10)

forms = import_formfile("Poisson2D.form")

a = forms.Poisson2DBilinearForm()
L = forms.Poisson2DLinearForm(f)

A = Matrix()
x = Vector()
b = Vector()

FEM_assemble(a, L, A, b, mesh, bc)

#print "A:"
#A.disp(False)

#print "b:"
#b.disp()

# Linear algebra could in certain cases be handled by Python modules,
# Numeric for example.

linearsolver = KrylovSolver()
linearsolver.solve(A, x, b)

#print "x:"
#x.disp()

trialelement = forms.Poisson2DBilinearFormTrialElement()
u = Function(x, mesh, trialelement)

vtkfile = File("poisson.vtk", File.vtk)
vtkfile << u

# Plotting can also be handled by Python modules (Mayavi for example)

