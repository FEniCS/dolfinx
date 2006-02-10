from dolfin import *
from math import *

eps = 1e-16

class ExactSolution(Function):
    def eval(self, point, i):
        if(i == 0):
            return -sin(pi * point.x) * cos(pi * point.y)
        else:
            return cos(pi * point.x) * sin(pi * point.y)

class Source(Function):
    def eval(self, point, i):
        if(i == 0):
            return -2.0 * pi * pi * sin(pi * point.x) * cos(pi * point.y)
        else:
            return 2.0 * pi * pi * cos(pi * point.x) * sin(pi * point.y)

class MyBC(BoundaryCondition):
    def eval(self, value, point, i):
        # Boundary condition for pressure
        if(i == 2):
            value.set(0.0)
            return

        # Boundary condition for velocity
        if(abs(point.x - 0.0) < eps):
            if(i == 0):
                value.set(0.0)
            else:
                value.set(sin(pi*point.y))
        elif(abs(point.x - 1.0) < eps):
            if(i == 0):
                value.set(0.0)
            else:
                value.set(-sin(pi*point.y))
        elif(abs(point.y - 0.0) < eps):
            if(i == 0):
                value.set(-sin(pi*point.x))
            else:
                value.set(0.0)
        elif(abs(point.y - 1.0) < eps):
            if(i == 0):
                value.set(sin(pi*point.x))
            else:
                value.set(0.0)


        return
    
f = Source()
bc = MyBC()
mesh = UnitSquare(16, 16)

forms = import_formfile("Stokes2D.form")

a = forms.Stokes2DBilinearForm()
L = forms.Stokes2DLinearForm(f)

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
linearsolver.setRtol(1.0e-15);
linearsolver.solve(A, x, b)

#print "x:"
#x.disp()

trialelement = a.trial()
w = Function(x, mesh, trialelement)

# Extract function slices
v = w[0]
p = w[1]


vfile = File("velocity.pvd")
pfile = File("pressure.pvd")

vfile << v
pfile << p

# Check error
u = ExactSolution()
l2errorform = import_formfile("L2Error.form")

l2errorL = l2errorform.L2ErrorLinearForm(v, u)

l2error = Vector()

FEM_assemble(l2errorL, l2error, mesh)

print "L2 error for velocity: ", l2error.norm()
