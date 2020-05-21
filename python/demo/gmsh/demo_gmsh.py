# Copyright (C) 2020 Garth N. Wells and Jørgen S. Dokken
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

from dolfinx import cpp
from dolfinx.cpp.io import cell_perm_vtk
from dolfinx.io import XDMFFile, ufl_mesh_from_gmsh
from dolfinx.mesh import create as create_mesh

# Generating a mesh on each process rank
# --------------------------------------
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
# ---------------------------------------------------------
#
# Generate mesh on rank 0, then build a distributed mesh

if MPI.COMM_WORLD.rank == 0:
    # Generate a mesh
    geom = pygmsh.opencascade.Geometry()
    geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
    pygmsh_mesh = pygmsh.generate_mesh(geom)

    # Extract the topology and geometry data
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points
    pygmsh_cell = pygmsh_mesh.cells[-1].type
    # Broadcast cell type data and geometric dimension
    cell_type, gdim, num_nodes = MPI.COMM_WORLD.bcast([pygmsh_cell, x.shape[1], cells.shape[1]], root=0)
else:
    cell_type, gdim, num_nodes = MPI.COMM_WORLD.bcast([None, None, None], root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, gdim])

mesh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh_from_gmsh(cell_type, gdim))
mesh.name = "ball_d1"
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as file:
    file.write_mesh(mesh)


# Create a distributed (parallel) mesh with quadratic geometry
# ------------------------------------------------------------
#
# Generate mesh on rank 0, then build a distributed mesh

if MPI.COMM_WORLD.rank == 0:
    geom = pygmsh.opencascade.Geometry()
    geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
    pygmsh_mesh = pygmsh.generate_mesh(geom, mesh_file_type="vtk", extra_gmsh_arguments=["-order", "2"])

    # Extract the topology and geometry data
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points
    pygmsh_cell = pygmsh_mesh.cells[-1].type

    # Broadcast cell type data and geometric dimension
    cell_type, gdim, num_nodes = MPI.COMM_WORLD.bcast([pygmsh_cell, x.shape[1], cells.shape[1]], root=0)
else:
    cell_type, gdim, num_nodes = MPI.COMM_WORLD.bcast([None, None, None], root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, gdim])

# Permute the topology from VTK to DOLFIN-X ordering
domain = ufl_mesh_from_gmsh(cell_type, gdim)
cell_type = cpp.mesh.to_type(str(domain.ufl_cell()))
cells = cells[:, cell_perm_vtk(cell_type, cells.shape[1])]

mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
mesh.name = "ball_d2"
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "a") as file:
    file.write_mesh(mesh)


if MPI.COMM_WORLD.rank == 0:
    # Generate a mesh with 2nd-order hexahedral cells using pygmsh
    lbw = [2, 3, 5]
    points = [geom.add_point([x, 0.0, 0.0], 1.0) for x in [0.0, lbw[0]]]
    line = geom.add_line(*points)
    _, rectangle, _ = geom.extrude(line, translation_axis=[0.0, lbw[1], 0.0], num_layers=lbw[1], recombine=True)
    geom.extrude(rectangle, translation_axis=[0.0, 0.0, lbw[2]], num_layers=lbw[2], recombine=True)
    pygmsh_mesh = pygmsh.generate_mesh(geom, mesh_file_type="vtk", extra_gmsh_arguments=["-order", "2"])

    # Extract the topology and geometry data
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points
    pygmsh_cell = pygmsh_mesh.cells[-1].type

    # Broadcast cell type data and geometric dimension
    cell_type, gdim, num_nodes = MPI.COMM_WORLD.bcast([pygmsh_cell, x.shape[1], cells.shape[1]], root=0)
else:
    # Receive cell type data and geometric dimension
    cell_type, gdim, num_nodes = MPI.COMM_WORLD.bcast([None, None, None], root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, gdim])

# Permute the mesh topology from VTK ordering to DOLFIN-X ordering
domain = ufl_mesh_from_gmsh(cell_type, gdim)
cell_type = cpp.mesh.to_type(str(domain.ufl_cell()))
cells = cells[:, cell_perm_vtk(cell_type, cells.shape[1])]

mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
mesh.name = "hex_d2"
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "a") as file:
    file.write_mesh(mesh)
