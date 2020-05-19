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
from dolfinx.cpp.io import permutation_vtk_to_dolfin, permute_cell_ordering
from dolfinx.io import XDMFFile, VTKFile
from dolfinx.mesh import create as create_mesh


def dolfin_to_gmsh(gmsh_cell):
    if gmsh_cell == "tetra":
        return
    elif gmsh_cell == "tetra10":
        return
    elif gmsh_cell == "hexahedron":
        return
    elif gmsh_cell == "hexahedron27":
        return
    elif gmsh_cell == "triangle":
        return [0, 1, 2]
    elif gmsh_cell == "triangle6":
        return [0, 1, 2, 5, 3, 4]
    elif gmsh_cell == "triangle10":
        # NOTE: Cannot be visualized in XDMFFile due to lack of XDMF
        # topology type interpreter at:
        # https://gitlab.kitware.com/vtk/vtk/blob/77104d45b773506a61586a6678e31864ab96676f/XdmfTopologyType.cpp#L184
        return [0, 1, 2, 7, 8, 3, 4, 6, 5, 9]
    elif gmsh_cell == "quad":
        return [0, 2, 3, 1]
    elif gmsh_cell == "quad9":
        return [0, 3, 4, 1, 6, 5, 7, 2, 8]
    elif gmsh_cell == "quad16":
        # NOTE: Same issues with paraview as Triangle1-
        # NOTE: according to the documentation, I would have thought
        # it should have been
        # https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
        # [0, 4, 5, 1, 8, 12, 6, 7, 13, 9, 2, 3, 10, 14, 15, 11]
        return [0, 4, 5, 1, 8, 12, 6, 7, 13, 9, 3, 2, 10, 14, 15, 11]
    else:
        raise RuntimeError("gmsh cell type '{}' not recognised".
                           format(gmsh_cell))


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
    elif gmsh_cell == "triangle":
        cell_shape = "triangle"
        degree = 1
    elif gmsh_cell == "triangle6":
        cell_shape = "triangle"
        degree = 2
    elif gmsh_cell == "triangle10":
        cell_shape = "triangle"
        degree = 3
    elif gmsh_cell == "quad":
        cell_shape = "quadrilateral"
        degree = 1
    elif gmsh_cell == "quad9":
        cell_shape = "quadrilateral"
        degree = 2
    elif gmsh_cell == "quad16":
        cell_shape = "quadrilateral"
        degree = 3
    else:
        raise RuntimeError("gmsh cell type '{}' not recognised".
                           format(gmsh_cell))

    cell = ufl.Cell(cell_shape, geometric_dimension=gdim)
    return ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))


def mesh_2D(quad, order):
    geom = pygmsh.opencascade.Geometry()
    geom.add_disk([0.0, 0.0, 0.0], 1.0, char_length=0.2)
    if quad:
        geom.add_raw_code("Recombine Surface {:};")
        geom.add_raw_code("Mesh.RecombinationAlgorithm = 2;")
    pygmsh_mesh = pygmsh.generate_mesh(geom,
                                       extra_gmsh_arguments=["-order", order])
    gmsh_celltype = pygmsh_mesh.cells[-1].type
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points
    cells_dolfin = permute_cell_ordering(cells, dolfin_to_gmsh(gmsh_celltype))
    mesh = create_mesh(MPI.COMM_SELF, cells_dolfin, x,
                       get_domain(gmsh_celltype, x.shape[1]))

    file = VTKFile("{}_mesh_rank_{}.pvd".format(gmsh_celltype,
                                                MPI.COMM_WORLD.rank))
    file.write(mesh)

    with XDMFFile(MPI.COMM_SELF, "{}_mesh_rank_{}.xdmf"
                  .format(gmsh_celltype, MPI.COMM_WORLD.rank), "w") as file:
        file.write_mesh(mesh)


for i in ["1", "2", "3"]:
    mesh_2D(False, i)
    print(i)
    mesh_2D(True, i)

# Generate a mesh on each rank with pygmsh, and create a DOLFIN-X mesh
# on each rank

geom = pygmsh.opencascade.Geometry()
geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
pygmsh_mesh = pygmsh.generate_mesh(geom)
cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points
gmsh_celltype = pygmsh_mesh.cells[-1].type
mesh = create_mesh(MPI.COMM_SELF, cells, x,
                   get_domain(gmsh_celltype, x.shape[1]))
with XDMFFile(MPI.COMM_SELF, "mesh_rank_{}.xdmf"
              .format(MPI.COMM_WORLD.rank), "w") as file:
    file.write_mesh(mesh)

# Generate mesh on rank 0, then build a distributed mesh
if MPI.COMM_WORLD.rank == 0:
    # Generate a mesh
    geom = pygmsh.opencascade.Geometry()
    geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
    pygmsh_mesh = pygmsh.generate_mesh(geom)

    # Extract the topology and geometry data
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points
    gmsh_celltype = pygmsh_mesh.cells[-1].type
    # Broadcast cell type data and geometric dimension
    cell_type, gdim, num_nodes = MPI.COMM_WORLD.bcast([gmsh_celltype,
                                                       x.shape[1],
                                                       cells.shape[1]], root=0)
else:
    cell_type, gdim, num_nodes = MPI.COMM_WORLD.bcast([None, None, None],
                                                      root=0)
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
    pygmsh_mesh = pygmsh.generate_mesh(geom, mesh_file_type="vtk",
                                       extra_gmsh_arguments=["-order", "2"])

    # Extract the topology and geometry data
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points

    # Broadcast cell type data and geometric dimension
    gmsh_celltype = pygmsh_mesh.cells[-1].type
    cell_type, gdim, num_nodes = MPI.COMM_WORLD.bcast([gmsh_celltype,
                                                       x.shape[1],
                                                       cells.shape[1]], root=0)
else:
    cell_type, gdim, num_nodes = MPI.COMM_WORLD.bcast([None, None, None],
                                                      root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, gdim])

# Permute the topology from VTK to DOLFIN-X ordering
domain = get_domain(cell_type, gdim)
cell_type = cpp.mesh.to_type(str(domain.ufl_cell()))
cells = cpp.io.permute_cell_ordering(cells,
                                     permutation_vtk_to_dolfin(cell_type,
                                                               cells.shape[1]))

mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
mesh.name = "ball_d2"
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "a") as file:
    file.write_mesh(mesh)


if MPI.COMM_WORLD.rank == 0:
    # Generate a mesh with 2nd-order hexahedral cells using pygmsh
    lbw = [2, 3, 5]
    points = [geom.add_point([x, 0.0, 0.0], 1.0) for x in [0.0, lbw[0]]]
    line = geom.add_line(*points)
    _, rectangle, _ = geom.extrude(line, translation_axis=[0.0, lbw[1], 0.0],
                                   num_layers=lbw[1], recombine=True)
    geom.extrude(rectangle, translation_axis=[0.0, 0.0, lbw[2]],
                 num_layers=lbw[2], recombine=True)
    pygmsh_mesh = pygmsh.generate_mesh(geom, mesh_file_type="vtk",
                                       extra_gmsh_arguments=["-order", "2"])

    # Extract the topology and geometry data
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points
    gmsh_celltype = pygmsh_mesh.cells[-1].type

    # Broadcast cell type data and geometric dimension
    cell_type, gdim, num_nodes = MPI.COMM_WORLD.bcast([gmsh_celltype,
                                                       x.shape[1],
                                                       cells.shape[1]], root=0)
else:
    # Receive cell type data and geometric dimension
    cell_type, gdim, num_nodes = MPI.COMM_WORLD.bcast([None, None, None],
                                                      root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, gdim])

# Permute the mesh topology from VTK ordering to DOLFIN-X ordering
domain = get_domain(cell_type, gdim)
cell_type = cpp.mesh.to_type(str(domain.ufl_cell()))
cells = cpp.io.permute_cell_ordering(cells,
                                     permutation_vtk_to_dolfin(cell_type,
                                                               cells.shape[1]))

mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
mesh.name = "hex_d2"
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "a") as file:
    file.write_mesh(mesh)
