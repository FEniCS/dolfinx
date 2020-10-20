from dolfin import *

mesh = Mesh()
with XDMFFile("mesh.xdmf") as xdmf:
    xdmf.read(mesh)
    mvc = MeshValueCollection('size_t', mesh, 3)
    xdmf.read(mvc, 'domains')
cells = cpp.mesh.MeshFunctionSizet(mesh, mvc)
