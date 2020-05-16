# Demonstrate the creation of meshes using pygmsh

import pygmsh
from mpi4py import MPI
import numpy as np

import ufl
from dolfinx import cpp
from dolfinx.io import XDMFFile
from dolfinx.mesh import create as create_mesh


def get_domain(gmsh_cell, gdim):
    if gmsh_cell == "tetra":
        cell_shape = "tetrahedron"
        degree = 1
    elif gmsh_cell == "tetra10":
        cell_shape = "tetrahedron"
        degree = 2
    else:
        raise RuntimeError("gmsh cell type not recognised")

    cell = ufl.Cell(cell_shape, geometric_dimension=gdim)
    return ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))


# Generate a mesh on each rank with pymsh, and create a DOLFIN-X mesh on
# each rank
geom = pygmsh.opencascade.Geometry()
geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
pygmsh_mesh = pygmsh.generate_mesh(geom)
pygmsh_mesh.prune()
cells, x = pygmsh_mesh.cells[0].data, pygmsh_mesh.points
mesh = create_mesh(MPI.COMM_SELF, cells, x, get_domain(pygmsh_mesh.cells[0].type, x.shape[1]))

# FIXME: Output using MPI.COMM_SELF gives an erro
# with XDMFFile(MPI.COMM_SELF, "mesh_rank0.xdmf", "w") as file:
#     file.write_mesh(mesh)


# Generate mesh on rank 0, then build a distributed mesh
if MPI.COMM_WORLD.rank == 0:
    geom = pygmsh.opencascade.Geometry()
    geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
    pygmsh_mesh = pygmsh.generate_mesh(geom)
    pygmsh_mesh.prune()
    cells, x = pygmsh_mesh.cells[0].data, pygmsh_mesh.points
    cell_type = MPI.COMM_WORLD.bcast(pygmsh_mesh.cells[0].type, root=0)
    gdim = MPI.COMM_WORLD.bcast(x.shape[1], root=0)
else:
    cell_type = MPI.COMM_WORLD.bcast(None, root=0)
    gdim = MPI.COMM_WORLD.bcast(None, root=0)
    cells, x = np.empty([0, 0]), np.empty([0, gdim])

mesh = create_mesh(MPI.COMM_WORLD, cells, x, get_domain(cell_type, gdim))
mesh.name = "ball_d1"
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as file:
    file.write_mesh(mesh)


# Generate mesh with quadratic geometry on rank 0, then build a
# distributed mesh
if MPI.COMM_WORLD.rank == 0:
    geom = pygmsh.opencascade.Geometry()
    geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
    pygmsh_mesh = pygmsh.generate_mesh(geom, mesh_file_type="vtk", extra_gmsh_arguments=["-order", "2"])
    pygmsh_mesh.prune()
    cells, x = pygmsh_mesh.cells[0].data, pygmsh_mesh.points
    cell_type = MPI.COMM_WORLD.bcast(pygmsh_mesh.cells[0].type, root=0)
    gdim = MPI.COMM_WORLD.bcast(x.shape[1], root=0)
else:
    cell_type = MPI.COMM_WORLD.bcast(None, root=0)
    gdim = MPI.COMM_WORLD.bcast(None, root=0)
    cells, x = np.empty([0, 0]), np.empty([0, gdim])

# Permute the topology from VTK to DOLFIN-X ordering
domain = get_domain(cell_type, gdim)
cell_type = cpp.mesh.to_type(str(domain.ufl_cell()))
cells = cpp.io.permute_cell_ordering(cells, cpp.io.permutation_vtk_to_dolfin(cell_type, cells.shape[1]))

mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
mesh.name = "ball_d2"
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "a") as file:
    file.write_mesh(mesh)
