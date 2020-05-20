# Copyright (C) 2020 Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
#
# Mesh generation for all cells and orders supported by Gmsh and pygmsh
# =====================================

import pygmsh
from mpi4py import MPI

from dolfinx.cpp.io import (permute_cell_ordering, permutation_gmsh_to_dolfin)
from dolfinx.io import XDMFFile, VTKFile, ufl_mesh_from_gmsh
from dolfinx.mesh import create as create_mesh


def mesh_2D(quad, order):
    """
    Function returning a mesh of a disk.
    Input:
       quad (bool): Mesh consisting of quadrilateral (True)
                    or triangular (False) cells
       order (str): "1", "2," or ", "3" describing the order for the cell.
    """
    geom = pygmsh.opencascade.Geometry()
    geom.add_disk([0.0, 0.0, 0.0], 1.0, char_length=0.2)
    if quad:
        geom.add_raw_code("Recombine Surface {:};")
        geom.add_raw_code("Mesh.RecombinationAlgorithm = 2;")
    pygmsh_mesh = pygmsh.generate_mesh(geom,
                                       extra_gmsh_arguments=["-order", order])
    gmsh_celltype = pygmsh_mesh.cells[-1].type
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points
    # Permute cells with msh ordering to dolfin node ordering
    cells_dolfin = permute_cell_ordering(cells,
                                         permutation_gmsh_to_dolfin(
                                             gmsh_celltype))

    mesh = create_mesh(MPI.COMM_SELF, cells_dolfin, x,
                       ufl_mesh_from_gmsh(gmsh_celltype, x.shape[1]))
    # Write mesh to file for visualization with VTK
    file = VTKFile("{}_mesh_rank_{}.pvd".format(gmsh_celltype,
                                                MPI.COMM_WORLD.rank))
    file.write(mesh)
    # Write mesh to file for visualization with XDMF
    with XDMFFile(MPI.COMM_SELF, "{}_mesh_rank_{}.xdmf"
                  .format(gmsh_celltype, MPI.COMM_WORLD.rank), "w") as file:
        file.write_mesh(mesh)


def mesh_tetra(order):
    """
    Returns a mesh consisting of tetrahedral cells of specified order.
    Mesh is a unit ball, subtracted the upper first quadrant.
    Input:
       order (str): "1", "2," describing the order for the cell.
    """
    geom = pygmsh.opencascade.Geometry()
    ball = geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
    box = geom.add_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom.boolean_difference([ball], [box])
    pygmsh_mesh = pygmsh.generate_mesh(geom,
                                       extra_gmsh_arguments=["-order", order])
    gmsh_celltype = pygmsh_mesh.cells[-1].type
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points
    # Permute cells with msh ordering to dolfin node ordering
    cells_dolfin = permute_cell_ordering(cells,
                                         permutation_gmsh_to_dolfin(
                                             gmsh_celltype))
    # Write mesh to file for visualization with VTK
    mesh = create_mesh(MPI.COMM_SELF, cells_dolfin, x,
                       ufl_mesh_from_gmsh(gmsh_celltype, x.shape[1]))

    file = VTKFile("{}_mesh_rank_{}.pvd".format(gmsh_celltype,
                                                MPI.COMM_WORLD.rank))
    file.write(mesh)
    # Write mesh to file for visualization with XDMF
    with XDMFFile(MPI.COMM_SELF, "{}_mesh_rank_{}.xdmf"
                  .format(gmsh_celltype, MPI.COMM_WORLD.rank), "w") as file:
        file.write_mesh(mesh)


def mesh_hex(order):
    """
    Returns a hexahedral mesh of specified order.
    Mesh is a cylinder.
    Input:
       order (str): "1", "2," describing the order for the cell.
    """
    geom = pygmsh.opencascade.Geometry()
    disk = geom.add_disk([0.0, 0.0, 0.0], 1.0, char_length=0.2)
    geom.add_raw_code("Mesh.RecombinationAlgorithm = 2;")
    geom.add_raw_code("Recombine Surface {:};")
    geom.extrude(disk, translation_axis=[0, 0, 1],
                 num_layers=2, recombine=True)

    pygmsh_mesh = pygmsh.generate_mesh(geom,
                                       extra_gmsh_arguments=["-order", order])
    gmsh_celltype = pygmsh_mesh.cells[-1].type
    # Permute cells with msh ordering to dolfin node ordering
    cells, x = pygmsh_mesh.cells[-1].data, pygmsh_mesh.points
    cells_dolfin = permute_cell_ordering(cells,
                                         permutation_gmsh_to_dolfin(
                                             gmsh_celltype))
    mesh = create_mesh(MPI.COMM_SELF, cells_dolfin, x,
                       ufl_mesh_from_gmsh(gmsh_celltype, x.shape[1]))
    # Write mesh to file for visualization with VTK
    file = VTKFile("{}_mesh_rank_{}.pvd".format(gmsh_celltype,
                                                MPI.COMM_WORLD.rank))
    file.write(mesh)
    # Write mesh to file for visualization with XDMF
    with XDMFFile(MPI.COMM_SELF, "{}_mesh_rank_{}.xdmf"
                  .format(gmsh_celltype, MPI.COMM_WORLD.rank), "w") as file:
        file.write_mesh(mesh)


# Generate all supported 2D meshes
for i in ["1", "2", "3"]:
    mesh_2D(False, i)
    mesh_2D(True, i)

# Generate tetrahedron meshes for all supported orders
for i in ["1", "2", "3"]:
    mesh_tetra(i)

# Generate hexahedron meshes for all supported orders
for i in ["1", "2"]:
    mesh_hex(i)
