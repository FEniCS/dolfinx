# Copyright (C) 2020 Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# =====================================
# Mesh generation using Gmsh and pygmsh
# =====================================

import numpy as np
import pygmsh
from mpi4py import MPI

from dolfinx import cpp
from dolfinx.cpp.io import perm_gmsh, extract_local_entities
from dolfinx.io import XDMFFile, ufl_mesh_from_gmsh
from dolfinx.mesh import create_mesh, create_meshtags

# Generating a mesh on each process rank
# ======================================
#
# Generate a mesh on each rank with pygmsh, and create a DOLFIN-X mesh
# on each rank

geom = pygmsh.opencascade.Geometry()
geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
pygmsh_mesh = pygmsh.generate_mesh(geom)
cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points
pygmsh_cell = pygmsh_mesh.cells[-1].type
mesh = create_mesh(MPI.COMM_SELF, cells, x,
                   ufl_mesh_from_gmsh(pygmsh_cell, x.shape[1]))

with XDMFFile(MPI.COMM_SELF, "mesh_rank_{}.xdmf".format(MPI.COMM_WORLD.rank), "w") as file:
    file.write_mesh(mesh)

# Create a distributed (parallel) mesh with affine geometry
# =========================================================
#
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

mesh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh_from_gmsh("tetra", 3))
mesh.name = "ball_d1"

local_entities, local_values = extract_local_entities(mesh, 2, marked_entities, values)
mesh.topology.create_connectivity(2, 0)
mt = create_meshtags(mesh, 2, cpp.graph.AdjacencyList_int32(local_entities), np.int32(local_values))
mt.name = "ball_d1_surface"

with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as file:
    file.write_mesh(mesh)

    mesh.topology.create_connectivity(2, 3)
    file.write_meshtags(mt, geometry_xpath="/Xdmf/Domain/Grid[@Name='ball_d1']/Geometry")

# Create a distributed (parallel) mesh with quadratic geometry
# ============================================================
#
# Generate mesh on rank 0, then build a distributed mesh

if MPI.COMM_WORLD.rank == 0:
    geom = pygmsh.opencascade.Geometry()
    ball = geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
    box = geom.add_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

    cut = geom.boolean_difference([ball], [box])
    geom.add_raw_code("Physical Surface(1) = {1};")
    geom.add_physical(cut, 2)

    pygmsh_mesh = pygmsh.generate_mesh(geom, extra_gmsh_arguments=["-order", "2"])

    # Extract the topology and geometry data
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points
    pygmsh_cell = pygmsh_mesh.cells[-1].type

    # Extract marked facets
    marked_entities = pygmsh_mesh.cells[-2].data
    values = pygmsh_mesh.cell_data["gmsh:physical"][-2]

    # Broadcast cell type data and geometric dimension
    num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
else:
    num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
    marked_entities, values = np.empty((0, 6)), np.empty((0,))

# Permute the topology from GMSH to DOLFIN-X ordering
domain = ufl_mesh_from_gmsh("tetra10", 3)
gmsh_tetra10 = perm_gmsh(cpp.mesh.CellType.tetrahedron, 10)
cells = cells[:, gmsh_tetra10]

mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
mesh.name = "ball_d2"

# Permute also entities which are tagged
gmsh_triangle6 = perm_gmsh(cpp.mesh.CellType.triangle, 6)
marked_entities = marked_entities[:, gmsh_triangle6]

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
    pygmsh_mesh = pygmsh.generate_mesh(geom, extra_gmsh_arguments=["-order", "2"])

    # Extract the topology and geometry data
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points
    pygmsh_cell = pygmsh_mesh.cells[-1].type

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

# Permute the mesh topology from GMSH ordering to DOLFIN-X ordering
domain = ufl_mesh_from_gmsh("hexahedron27", 3)
gmsh_hex27 = perm_gmsh(cpp.mesh.CellType.hexahedron, 27)
cells = cells[:, gmsh_hex27]

mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
mesh.name = "hex_d2"

# Permute also entities which are tagged
gmsh_quad9 = perm_gmsh(cpp.mesh.CellType.quadrilateral, 9)
marked_entities = marked_entities[:, gmsh_quad9]

local_entities, local_values = extract_local_entities(mesh, 2, marked_entities, values)
mesh.topology.create_connectivity(2, 0)
mt = create_meshtags(mesh, 2, cpp.graph.AdjacencyList_int32(local_entities), np.int32(local_values))
mt.name = "hex_d2_surface"

with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "a") as file:
    file.write_mesh(mesh)

    mesh.topology.create_connectivity(2, 3)
    file.write_meshtags(mt, geometry_xpath="/Xdmf/Domain/Grid[@Name='hex_d2']/Geometry")
