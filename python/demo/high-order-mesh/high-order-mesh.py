# Higher-order mesh with VTK visualization
from pygmsh.built_in import Geometry
from pygmsh import generate_mesh

import meshio

from dolfin import Mesh, cpp, MPI
from dolfin.io import VTKFile

# Generate high order gmsh
order = 2
geo = Geometry()
geo.add_raw_code("Mesh.ElementOrder={0};".format(order))
circle = geo.add_circle([0, 0, 0], 1, lcar=0.4)
geo.add_physical(circle.plane_surface, 1)

pygmsh = generate_mesh(geo, prune_z_0=True, verbose=False)

if order == 1:
    element = "triangle"
else:
    element = "triangle{0:d}".format(int((order + 1) * (order + 2) / 2))

meshio.write("mesh_meshio.vtk",
             meshio.Mesh(points=pygmsh.points,
                         cells={element: pygmsh.cells[element]}),
             file_format="vtk-ascii")


mesh = Mesh(MPI.comm_world, cpp.mesh.CellType.triangle, pygmsh.points,
            pygmsh.cells[element], [], cpp.mesh.GhostMode.none)
# mesh.create_connectivity(1, 2)
VTKFile("mesh.vtk").write(mesh)
