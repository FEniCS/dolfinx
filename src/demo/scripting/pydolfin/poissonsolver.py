from dolfin import *
from math import *

class Source(Function):
    def __call__(self, point):
#        return pi * pi * sin(pi * point.x)
        return point.y + 1.0

class SimpleBC(BoundaryCondition):
    def __call__(self, point):
        value = BoundaryValue()
        if point.x == 0.0 or point.x == 1.0:
            value.set(0.0)
        return value
    
f = Source()
bc = SimpleBC()
mesh = UnitSquare(10, 10)

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

vtkfile = File("poisson.vtk", File.vtk)
vtkfile << u

# Plot using Mayavi

import mayavi
v = mayavi.mayavi(geometry="743x504") # create a MayaVi window.
d = v.open_vtk_xml("poisson000001.vtu", config=0) # open the data file.
# The config option turns on/off showing a GUI control for the
# data/filter/module.

# Load the filters.
f = v.load_filter('WarpScalar', config=0)
n = v.load_filter('PolyDataNormals', 0)
n.fil.SetFeatureAngle(45) # configure the normals.

# Load the necessary modules.
m = v.load_module('SurfaceMap', 0)
a = v.load_module('Axes', 0)
a.axes.SetCornerOffset(0.0) # configure the axes module.
a.axes.SetFontFactor(1.0) # configure the axes module.
o = v.load_module('Outline', 0)

camera = v.renwin.camera
camera.Zoom(0.7)

# Re-render the scene.
v.Render()


def myanim(t, v, f):
    t[0] += 0.01
    f.fil.SetScaleFactor(1.5 * (sin(t[0]) + 1.2))
    v.Render()
#    v.renwin.save_png('/tmp/anim%f.png' % t[0])
    v.renwin.save_png('anim%4.4f.png' % t[0])

t = [0.0] # wrap the float in a mutable object
v.start_animation(10, myanim, t, v, f)

