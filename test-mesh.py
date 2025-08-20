import gmsh
import sys

gmsh.initialize()
gmsh.model.add("third_order_hexahedron")

# Create a box (1x1x1 cube)
box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)

# Synchronize CAD kernel
gmsh.model.occ.synchronize()

# Set third-order mesh
gmsh.option.setNumber("Mesh.ElementOrder", 3)

# To create a single Hex27 element, we use transfinite meshing:
# Get volume, surfaces, and edges to apply transfinite constraints
surfaces = gmsh.model.getEntities(2)
edges = gmsh.model.getEntities(1)
volume = gmsh.model.getEntities(3)[0][1]

# Apply transfinite curves (edges)
for dim, tag in edges:
    gmsh.model.mesh.setTransfiniteCurve(tag, 3)  # 3 nodes per edge: 2 ends + 1 internal (3rd order)

# Apply transfinite surfaces
for dim, tag in surfaces:
    gmsh.model.mesh.setTransfiniteSurface(tag)

# Apply transfinite volume
gmsh.model.mesh.setTransfiniteVolume(volume)

# Recombine into hexahedra
for dim, tag in surfaces:
    gmsh.model.mesh.setRecombine(dim, tag)

gmsh.model.addPhysicalGroup(3, [volume])

# Generate 3D mesh
gmsh.model.mesh.generate(3)

# Optional: write to file
# gmsh.write("third_order_hexahedron.msh")

# Launch GUI
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

import dolfinx
from mpi4py import MPI
mesh_data = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0)

with dolfinx.io.XDMFFile(MPI.COMM_SELF, "test-mesh.xdmf", "w", dolfinx.io.XDMFFile.Encoding.ASCII) as file:
    file.write_mesh(mesh_data.mesh)


# gmsh.finalize()
