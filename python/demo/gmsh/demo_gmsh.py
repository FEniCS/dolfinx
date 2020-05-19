# Copyright (C) 2020 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
#
# Mesh generation using Gmsh and pygmsh
# =====================================

import numpy as np
import pygmsh
from mpi4py import MPI

import ufl
from dolfinx import cpp
from dolfinx.cpp.io import permutation_vtk_to_dolfin, extract_local_entities
from dolfinx.io import XDMFFile
from dolfinx.mesh import create as create_mesh
from dolfinx.cpp.mesh import create_meshtags
from dolfinx.fem import create_coordinate_map


def get_domain(gmsh_cell, gdim):
    if gmsh_cell == "tetra":
        cell_shape = "tetrahedron"
        degree = 1
    elif gmsh_cell == "tetra10":
        cell_shape = "tetrahedron"
        degree = 2
    elif gmsh_cell == "hexahedron":
        cell_shape = "hexahedron"
        degree = 1
    elif gmsh_cell == "hexahedron27":
        cell_shape = "hexahedron"
        degree = 2
    elif gmsh_cell == "line3":
        cell_shape = "interval"
        degree = 2
    else:
        raise RuntimeError("gmsh cell type '{}' not recognised".format(gmsh_cell))

    cell = ufl.Cell(cell_shape, geometric_dimension=gdim)
    return ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))


# Generate a mesh on each rank with pygmsh, and create a DOLFIN-X mesh
# on each rank
geom = pygmsh.opencascade.Geometry()
geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
pygmsh_mesh = pygmsh.generate_mesh(geom)
cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points
mesh = create_mesh(MPI.COMM_SELF, cells, x, get_domain(pygmsh_mesh.cells[-1].type, x.shape[1]))

with XDMFFile(MPI.COMM_SELF, "mesh_rank_{}.xdmf".format(MPI.COMM_WORLD.rank), "w") as file:
    file.write_mesh(mesh)

# Generate mesh on rank 0, then build a distributed mesh
if MPI.COMM_WORLD.rank == 0:
    # Generate a mesh
    geom = pygmsh.opencascade.Geometry()
    geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
    pygmsh_mesh = pygmsh.generate_mesh(geom)

    # Extract the topology and geometry data
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points

    # Broadcast cell type data and geometric dimension
    cell_type, gdim, num_nodes = MPI.COMM_WORLD.bcast([pygmsh_mesh.cells[-1].type, x.shape[1],
                                                       cells.shape[1]], root=0)
else:
    cell_type, gdim, num_nodes = MPI.COMM_WORLD.bcast([None, None, None], root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, gdim])

mesh = create_mesh(MPI.COMM_WORLD, cells, x, get_domain(cell_type, gdim))
mesh.name = "ball_d1"
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as file:
    file.write_mesh(mesh)


# Generate mesh with quadratic geometry on rank 0, then build a
# distributed mesh
if MPI.COMM_WORLD.rank == 0:
    geom = pygmsh.opencascade.Geometry()
    geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
    pygmsh_mesh = pygmsh.generate_mesh(geom, mesh_file_type="msh", extra_gmsh_arguments=["-order", "2"])

    # Extract the topology and geometry data
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points

    # Broadcast cell type data and geometric dimension
    cell_type, gdim, num_nodes = MPI.COMM_WORLD.bcast([pygmsh_mesh.cells[-1].type, x.shape[1],
                                                       cells.shape[1]], root=0)
else:
    cell_type, gdim, num_nodes = MPI.COMM_WORLD.bcast([None, None, None], root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, gdim])

# Permute the topology from VTK to DOLFIN-X ordering
domain = get_domain(cell_type, gdim)
cell_type = cpp.mesh.to_type(str(domain.ufl_cell()))
cells = cpp.io.permute_cell_ordering(cells, permutation_vtk_to_dolfin(cell_type, cells.shape[1]))

mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
mesh.name = "ball_d2"
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "a") as file:
    file.write_mesh(mesh)


if MPI.COMM_WORLD.rank == 0:
    # Generate a mesh with 2nd-order hexahedral cells using pygmsh
    geom = pygmsh.opencascade.Geometry()
    circle = geom.add_disk([0.0, 0.0, 0.0], 1.0)
    geom.add_raw_code("Recombine Surface {%s};" % circle.id)

    _, box, _ = geom.extrude(circle, translation_axis=[0.0, 0.0, 5], num_layers=5, recombine=True)

    geom.add_physical(circle, label=1)
    geom.add_physical(box, label=2)

    pygmsh_mesh = pygmsh.generate_mesh(geom, mesh_file_type="msh", extra_gmsh_arguments=["-order", "2"])

    # Extract the topology and geometry data
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points

    # Extract marked line
    marked_entities = pygmsh_mesh.cells[-2].data
    values = pygmsh_mesh.cell_data["gmsh:physical"][-2]

    # Broadcast cell type data and geometric dimension
    num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
else:
    # Receive cell type data and geometric dimension
    num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
    cells, x = np.empty((0, num_nodes)), np.empty((0, 3))
    marked_entities, values = np.empty((0, 9)), np.empty((0,))

gmsh_to_vtk = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19,
                        17, 10, 12, 14, 15, 22, 23, 21, 24, 20, 25, 26])
vtk_to_gmsh = np.argsort(gmsh_to_vtk)

# Permute the mesh topology from VTK ordering to DOLFIN-X ordering
cell_type = "hexahedron27"
domain = get_domain(cell_type, 3)
cell_type = cpp.mesh.to_type(str(domain.ufl_cell()))
dolfin_to_vtk = permutation_vtk_to_dolfin(cell_type, cells.shape[1])

cells = cpp.io.permute_cell_ordering(cells, vtk_to_gmsh)
cells = cpp.io.permute_cell_ordering(cells, dolfin_to_vtk)

mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
mesh.name = "hex_d2"

# Construct the coordinate map
cell = ufl.Cell("quadrilateral", geometric_dimension=3)
domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 2))
cmap = create_coordinate_map(domain)

quad9 = permutation_vtk_to_dolfin(cpp.mesh.to_type("quadrilateral"), 9)
marked_entities = cpp.io.permute_cell_ordering(marked_entities, quad9)
local_entities, local_values = extract_local_entities(mesh, cmap, marked_entities, values)

mt = create_meshtags(mesh, 2, cpp.graph.AdjacencyList_int32(local_entities), np.int32(local_values))
mt.name = "tags"

with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "a") as file:
    file.write_mesh(mesh)
    mesh.topology.create_connectivity_all()
    file.write_meshtags(mt)
