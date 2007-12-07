from dolfin import *

mesh = Mesh("mesh2D.xml.gz")
mesh_function = MeshFunction("real", mesh, "meshfunction.xml")
plot(mesh_function)
