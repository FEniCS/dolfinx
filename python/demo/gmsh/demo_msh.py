# Copyright (C) 2020 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
#
# Mesh generation using Gmsh and pygmsh
# =====================================

import pygmsh
from mpi4py import MPI

import ufl
from dolfinx.io import gmsh_to_dolfin, ufl_mesh_from_gmsh
from dolfinx.cpp.io import permute_cell_ordering
from dolfinx.io import XDMFFile, VTKFile
from dolfinx.mesh import create as create_mesh


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
    cells_dolfin = permute_cell_ordering(cells, gmsh_to_dolfin(gmsh_celltype))

    mesh = create_mesh(MPI.COMM_SELF, cells_dolfin, x,
                       ufl_mesh_from_gmsh(gmsh_celltype, x.shape[1]))
    file = VTKFile("{}_mesh_rank_{}.pvd".format(gmsh_celltype,
                                                MPI.COMM_WORLD.rank))
    file.write(mesh)

    with XDMFFile(MPI.COMM_SELF, "{}_mesh_rank_{}.xdmf"
                  .format(gmsh_celltype, MPI.COMM_WORLD.rank), "w") as file:
        file.write_mesh(mesh)


for i in ["1", "2", "3"]:
    mesh_2D(False, i)
    mesh_2D(True, i)


def mesh_tetra(order):
    geom = pygmsh.opencascade.Geometry()
    geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
    pygmsh_mesh = pygmsh.generate_mesh(geom,
                                       extra_gmsh_arguments=["-order", order])
    gmsh_celltype = pygmsh_mesh.cells[-1].type
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points
    cells_dolfin = permute_cell_ordering(cells, gmsh_to_dolfin(gmsh_celltype))
    mesh = create_mesh(MPI.COMM_SELF, cells_dolfin, x,
                       ufl_mesh_from_gmsh(gmsh_celltype, x.shape[1]))

    file = VTKFile("{}_mesh_rank_{}.pvd".format(gmsh_celltype,
                                                MPI.COMM_WORLD.rank))
    file.write(mesh)

    with XDMFFile(MPI.COMM_SELF, "{}_mesh_rank_{}.xdmf"
                  .format(gmsh_celltype, MPI.COMM_WORLD.rank), "w") as file:
        file.write_mesh(mesh)


def mesh_hex(order):
    geom = pygmsh.opencascade.Geometry()
    rect = geom.add_rectangle([0, 0, 0], 1, 1, char_length=1)
    geom.add_raw_code("Mesh.RecombinationAlgorithm = 2;")
    geom.add_raw_code("Recombine Surface {:};")
    geom.extrude(rect, translation_axis=[0, 0, 1],
                 num_layers=1, recombine=True)

    pygmsh_mesh = pygmsh.generate_mesh(geom,
                                       extra_gmsh_arguments=["-order", order])
    gmsh_celltype = pygmsh_mesh.cells[-1].type
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points
    cells_dolfin = permute_cell_ordering(cells, gmsh_to_dolfin(gmsh_celltype))
    mesh = create_mesh(MPI.COMM_SELF, cells_dolfin, x,
                       ufl_mesh_from_gmsh(gmsh_celltype, x.shape[1]))

    file = VTKFile("{}_mesh_rank_{}.pvd".format(gmsh_celltype,
                                                MPI.COMM_WORLD.rank))
    file.write(mesh)

    with XDMFFile(MPI.COMM_SELF, "{}_mesh_rank_{}.xdmf"
                  .format(gmsh_celltype, MPI.COMM_WORLD.rank), "w") as file:
        file.write_mesh(mesh)


for i in ["1", "2", "3"]:
    mesh_tetra(i)
for i in ["1", "2"]:
    mesh_hex(i)
