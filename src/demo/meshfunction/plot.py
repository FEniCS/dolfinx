from dolfin import *

mesh = Mesh("mesh2D.xml.gz")
mesh_function = MeshFunction("real", mesh, "mfv.xml")
plot(mesh_function)
