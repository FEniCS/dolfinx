from dolfin import *
from math import *
import ffc.compiler.compiler as ffc

# Define right-hand side
class MyFunction(Function):
    def eval(self, point, i):
        return 0.1 * sin(20.0 * pow(point[0], 3.0) * point[1])

f = MyFunction()

# Create a mesh of the unit square
mesh = UnitSquare(60, 60)

K = ffc.FiniteElement("Lagrange", "triangle", 1)

Pforms = projection(K, "Projection")

a = Pforms.ProjectionBilinearForm()
L = Pforms.ProjectionLinearForm(f)

# Assemble linear system
M = Matrix()
b = Vector()
FEM_assemble(a, L, M, b, mesh)

# Solve linear system
m = Vector()
FEM_lump(M, m)

x = Vector(b.size())
x.copy(b, 0, 0, x.size())
x.div(m)

# Define a function from computed degrees of freedom
u = Function(x, mesh, a.trial())

# Save solution to file in VTK format
file = File("projection.pvd")
file << u

# Plot with Mayavi

# Load mayavi
from mayavi import *

# Plot solution
v = mayavi()
d = v.open_vtk_xml("projection000000.vtu")
m = v.load_module("Axes")
m = v.load_module("BandedSurfaceMap")
fi = v.load_filter('WarpScalar', config=0)

camera = v.renwin.camera
camera.Zoom(1.0)
#camera.SetPosition(1.0, 3.0, -5.0)
#camera.SetFocalPoint(0.0, 0.0, 0.0)
#camera.SetRoll(0.0)

v.renwin.Render()

# Wait until window is closed
v.master.wait_window()
