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
from dolfinx.cpp.io import extract_local_entities
from dolfinx.io import XDMFFile
from dolfinx.mesh import create as create_mesh
from dolfinx.cpp.mesh import create_meshtags


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
    ball = geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
    box = geom.add_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

    cut = geom.boolean_difference([ball], [box])
    geom.add_raw_code("Physical Surface(1) = {1};")
    geom.add_physical(cut, 2)

    pygmsh_mesh = pygmsh.generate_mesh(geom)

    # Extract the topology and geometry data
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points

    # Extract marked facets
    marked_entities = pygmsh_mesh.cells[-2].data
    values = pygmsh_mesh.cell_data["gmsh:physical"][-2]

    # Broadcast cell type data and geometric dimension
    num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
else:
    num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
    marked_entities, values = np.empty((0, 3)), np.empty((0,))


domain = get_domain("tetra", 3)
gmsh_to_dolfin_tet = np.array([0, 1, 2, 3])
cells = cpp.io.permute_cell_ordering(cells, gmsh_to_dolfin_tet)

mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
mesh.name = "ball_d1"

gmsh_to_dolfin_tri = np.array([0, 1, 2])
marked_entities = cpp.io.permute_cell_ordering(marked_entities, gmsh_to_dolfin_tri)

local_entities, local_values = extract_local_entities(mesh, 2, marked_entities, values)
mesh.topology.create_connectivity(2, 0)
mt = create_meshtags(mesh, 2, cpp.graph.AdjacencyList_int32(local_entities), np.int32(local_values))
mt.name = "ball_d1_surface"

with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as file:
    file.write_mesh(mesh)

    mesh.topology.create_connectivity(2, 3)
    file.write_meshtags(mt, geometry_xpath="/Xdmf/Domain/Grid[@Name='ball_d1']/Geometry")

# Generate mesh with quadratic geometry on rank 0, then build a
# distributed mesh
if MPI.COMM_WORLD.rank == 0:
    geom = pygmsh.opencascade.Geometry()
    ball = geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
    box = geom.add_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

    cut = geom.boolean_difference([ball], [box])
    geom.add_raw_code("Physical Surface(1) = {1};")
    geom.add_physical(cut, 2)

    pygmsh_mesh = pygmsh.generate_mesh(geom, mesh_file_type="msh", extra_gmsh_arguments=["-order", "2"])

    # Extract the topology and geometry data
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points

    # Extract marked facets
    marked_entities = pygmsh_mesh.cells[-2].data
    values = pygmsh_mesh.cell_data["gmsh:physical"][-2]

    # Broadcast cell type data and geometric dimension
    num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
else:
    num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
    marked_entities, values = np.empty((0, 6)), np.empty((0,))

# Permute the topology from VTK to DOLFIN-X ordering
domain = get_domain("tetra10", 3)
gmsh_to_dolfin_tet10 = np.array([0, 1, 2, 3, 9, 6, 8, 7, 5, 4])
cells = cpp.io.permute_cell_ordering(cells, gmsh_to_dolfin_tet10)

mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
mesh.name = "ball_d2"

gmsh_to_dolfin_tri6 = np.array([0, 1, 2, 5, 3, 4])
marked_entities = cpp.io.permute_cell_ordering(marked_entities, gmsh_to_dolfin_tri6)

local_entities, local_values = extract_local_entities(mesh, 2, marked_entities, values)
mesh.topology.create_connectivity(2, 0)
mt = create_meshtags(mesh, 2, cpp.graph.AdjacencyList_int32(local_entities), np.int32(local_values))
mt.name = "ball_d2_surface"

with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "a") as file:
    file.write_mesh(mesh)

    mesh.topology.create_connectivity(2, 3)
    file.write_meshtags(mt, geometry_xpath="/Xdmf/Domain/Grid[@Name='ball_d2']/Geometry")


if MPI.COMM_WORLD.rank == 0:
    # Generate a mesh with 2nd-order hexahedral cells using pygmsh
    geom = pygmsh.opencascade.Geometry()
    geom.add_raw_code("Mesh.RecombineAll = 1;")
    geom.add_raw_code("Mesh.CharacteristicLengthFactor = 1.0;")
    geom.add_raw_code("Mesh.RecombinationAlgorithm = 2;")
    circle = geom.add_disk([0.0, 0.0, 0.0], 1.0)
    circle_inner = geom.add_disk([0.0, 0.0, 0.0], 0.5)

    cut = geom.boolean_difference([circle], [circle_inner])
    _, box, _ = geom.extrude(cut, translation_axis=[0.0, 0.0, 5], num_layers=5, recombine=True)

    geom.add_physical(cut, label=1)
    geom.add_physical(box, label=2)
    pygmsh_mesh = pygmsh.generate_mesh(geom, mesh_file_type="msh", extra_gmsh_arguments=["-order", "2"])

    # Extract the topology and geometry data
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points

    # Extract marked facets
    marked_entities = pygmsh_mesh.cells[-2].data
    values = pygmsh_mesh.cell_data["gmsh:physical"][-2]

    # Broadcast cell type data and geometric dimension
    num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
else:
    # Receive cell type data and geometric dimension
    num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
    cells, x = np.empty((0, num_nodes)), np.empty((0, 3))
    marked_entities, values = np.empty((0, 9)), np.empty((0,))

gmsh_to_dolfin_hex27 = np.array([0, 9, 12, 3, 1, 10, 13, 4, 18, 6, 2, 15, 11, 21, 14,
                                 5, 19, 7, 16, 22, 24, 20, 8, 17, 23, 25, 26])

# Permute the mesh topology from VTK ordering to DOLFIN-X ordering
cell_type = "hexahedron27"
domain = get_domain(cell_type, 3)
cells = cpp.io.permute_cell_ordering(cells, gmsh_to_dolfin_hex27)

mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
mesh.name = "hex_d2"

gmsh_to_dolfin_quad9 = np.array([0, 3, 4, 1, 6, 5, 7, 2, 8])
marked_entities = cpp.io.permute_cell_ordering(marked_entities, gmsh_to_dolfin_quad9)

local_entities, local_values = extract_local_entities(mesh, 2, marked_entities, values)
mesh.topology.create_connectivity(2, 0)
mt = create_meshtags(mesh, 2, cpp.graph.AdjacencyList_int32(local_entities), np.int32(local_values))
mt.name = "hex_d2_surface"

with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "a") as file:
    file.write_mesh(mesh)

    mesh.topology.create_connectivity(2, 3)
    file.write_meshtags(mt, geometry_xpath="/Xdmf/Domain/Grid[@Name='hex_d2']/Geometry")
