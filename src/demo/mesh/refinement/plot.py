from dolfin import *

mesh = Mesh("mesh.xml")
plot(mesh)

mesh_refined = Mesh("mesh_refined.xml")
plot(mesh_refined)

mesh_coarsened = Mesh("mesh_coarsened.xml")
plot(mesh_coarsened)

mesh_refined = Mesh("mesh_boundary.xml")
plot(mesh_refined)
