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

linearsolver = KrylovSolver()
linearsolver.solve(A, x, b)

trialelement = forms.Poisson2DBilinearFormTrialElement()
u = Function(x, mesh, trialelement)

xmlfile = File("poisson.xml", File.xml)
xmlfile << u

