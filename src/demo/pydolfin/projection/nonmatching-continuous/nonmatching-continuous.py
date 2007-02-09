from dolfin import *

# Projection demo between non-matching meshes, continuous Lagrange

meshA = Mesh()
editor = MeshEditor()
editor.open(meshA, "triangle", 2, 2)
editor.initVertices(4)
editor.addVertex(0, 0.0, 0.0)
editor.addVertex(1, 1.0, 0.0)
editor.addVertex(2, 1.0, 1.0)
editor.addVertex(3, 0.0, 1.0)
editor.initCells(2)
editor.addCell(0, 0, 1, 3)
editor.addCell(1, 1, 2, 3)
editor.close()

meshA.refine()
meshA.refine()
meshA.refine()
#meshA.refine()
#meshA.refine()
#meshA.refine()
#meshA.refine()
#meshA.refine()

NA = meshA.numVertices()
xA = Vector(NA)

for v in vertices(meshA):
    id = v.index()
    p = v.point()
    xA[id] = (p[1] - 0.5) * pow(p[0] - 0.5, 2)

K = P1tri()

fA = Function(xA, meshA, K)

meshB = UnitSquare(11, 11)
fN = Function(fA, meshB)
fN.attach(meshB)

fileA = File("fA.pvd")
fileA << fA

NB = meshB.numVertices()
xB = Vector(NB)

fB = Function(xB, meshB, K)

forms = import_formfile("Projection.form")

a = forms.ProjectionBilinearForm()
L = forms.ProjectionLinearForm(fN)

print "element:"
print a.trial().spec().repr()

A = Matrix()
b = Vector()

FEM_assemble(a, A, meshB)
FEM_assemble(L, b, meshB)

Alump = Vector(b.size())

#print "A:"
#A.disp()
#print "b:"
#b.disp()


solver = KrylovSolver()
solver.solve(A, xB, b)
#xB.copy(b, 0, 0, b.size())
#A.lump(Alump)
#xB.div(Alump)

print "xB:"
xB.disp()

fileB = File("fB.pvd")
fileB << fB

# Plot with Mayavi

# Load mayavi
from mayavi import *

# Plot solution
v = mayavi()
dA = v.open_vtk_xml("fA000000.vtu")
mA = v.load_module("BandedSurfaceMap")
fiA = v.load_filter('WarpScalar', config=0)

dB = v.open_vtk_xml("fB000000.vtu")
mB = v.load_module("BandedSurfaceMap")
fiB = v.load_filter('WarpScalar', config=0)

# Wait until window is closed
v.master.wait_window()
