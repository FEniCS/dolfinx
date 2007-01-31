from dolfin import *
from math import *

eps = 1e-16

class ExactSolution(Function):
    def eval(self, point, i):
        if(i == 0):
            return -sin(pi * point[0]) * cos(pi * point[1])
        else:
            return cos(pi * point[0]) * sin(pi * point[1])

class Source(Function):
    def eval(self, point, i):
        if(i == 0):
            return -2.0 * pi * pi * sin(pi * point[0]) * cos(pi * point[1])
        else:
            return 2.0 * pi * pi * cos(pi * point[0]) * sin(pi * point[1])

class MyBC(BoundaryCondition):
    def eval(self, value, point, i):
        # Boundary condition for pressure
        if(i == 2):
            value.set(0.0)
            return

        # Boundary condition for velocity
        if(abs(point[0] - 0.0) < eps):
            if(i == 0):
                value.set(0.0)
            else:
                value.set(sin(pi*point[1]))
        elif(abs(point[0] - 1.0) < eps):
            if(i == 0):
                value.set(0.0)
            else:
                value.set(-sin(pi*point[1]))
        elif(abs(point[1] - 0.0) < eps):
            if(i == 0):
                value.set(-sin(pi*point[0]))
            else:
                value.set(0.0)
        elif(abs(point[1] - 1.0) < eps):
            if(i == 0):
                value.set(sin(pi*point[0]))
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

#linearsolver = KrylovSolver()
linearsolver = LUSolver()

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

# Load mayavi
from mayavi import *

# Plot solution
v = mayavi()
d = v.open_vtk_xml("velocity000000.vtu")
m = v.load_module("VelocityVector", config=1)

# Wait until window is closed
v.master.wait_window()
