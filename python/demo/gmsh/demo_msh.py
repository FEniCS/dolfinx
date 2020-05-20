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
# from dolfinx import cpp
# from dolfinx.cpp.io import permutation_vtk_to_dolfin, permutation_dolfin_to_vtk
from dolfinx.cpp.io import permute_cell_ordering
from dolfinx.io import XDMFFile, VTKFile
from dolfinx.mesh import create as create_mesh


def dolfin_to_gmsh(gmsh_cell):
    """
    Returns the cell permutation from a GMSH local node ordering to the dolfinx node
    ordering.
    For triangle, the msh ordering is described at: 
    https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
    The dolfin-ordering is:
    Triangle:               Triangle6:          Triangle10: 
    v
    ^                                             
    |                                             
    2                       2                    2  
    |`\                     |`\                  | \ 
    |  `\                   |  `\                6   4
    |    `\                 4    `3              |     \
    |      `\               |      `\            5   9   3
    |        `\             |        `\          |         \
    0----------1 --> u      0-----5----1         0---7---8---1

    Quadrilateral:         Quadrilateral9:         Quadrilateral16:
    v
    ^
    |
    1-----------3          1-----7-----4           1---9--13---5
    |           |          |           |           |           |
    |           |          |           |           3  11  15   7
    |           |          2     8     5           |           |
    |           |          |           |           2  10  14   6
    |           |          |           |           |           |
    0-----------2 --> u    0-----6-----3           0---8--12---4

    Tetrahedron:                          Tetrahedron10:                         Tetrahedron20
                      v
                    .
                  ,/
                 /
              2                                     2                                     2 
            ,/|`\                                 ,/|`\                                 ,/|`\
          ,/  |  `\                             ,/  |  `\                             13  |  `9
         ,/    '.   `\                         ,8    '.   `6                        ,/     4   `\
       ,/       |     `\                     ,/       5     `\                     12    19 |     `8
     ,/         |       `\                 ,/         |       `\                 ,/         |       `\
    0-----------'.--------1 --> u         0--------9--'.--------1               0-----14----'.--15----1
     `\.         |      ,/                 `\.         |      ,/                 `\.  17     |  16  ,/
        `\.      |    ,/                      `\.      |    ,4                      10.   18 5    ,6
           `\.   '. ,/                           `7.   '. ,/                           `\.   '.  7
              `\. |/                                `\. |/                                11. |/
                 `3                                    `3                                    `3
                    `\.
                       ` w

    Hexahedron:          Hexahedron27:
           v
    2----------6           3----21----12
    |\     ^   |\          |\         |\
    | \    |   | \         | 5    23  | 14
    |  \   |   |  \        6  \ 24    15 \
    |   3------+---7       |   4----22+---13
    |   |  +-- |-- | -> u  | 8 |  26  | 17|
    0---+---\--4   |       0---+18----9   |
     \  |    \  \  |        \  7     25\ 16
      \ |     \  \ |         2 |   20   11|
       \|      w  \|          \|         \|
        1----------5           1----19----10

    """
    if gmsh_cell == "tetra":
        return [0, 1, 2, 3]
    elif gmsh_cell == "tetra10":
        return [0, 1, 2, 3, 9, 6, 8, 7, 4, 5]
    elif gmsh_cell == "tetra20":
        # NOTE: Cannot be visualized in XDMFFile due to lack of XDMF
        # topology type interpreter at:
        # https://gitlab.kitware.com/vtk/vtk/blob/77104d45b773506a61586a6678e31864ab96676f/XdmfTopologyType.cpp#L184
        return [0, 1, 2, 3, 14, 15, 8, 9, 13, 12, 11,
                10, 5, 4, 7, 6, 19, 18, 17, 16]
    elif gmsh_cell == "hexahedron":
        return [0, 4, 6, 2, 1, 5, 7, 3]
    elif gmsh_cell == "hexahedron27":
        return [0, 9, 12, 3, 1, 10, 13, 4, 18, 6, 2, 15, 11, 21, 14,
                5, 19, 7, 16, 22, 24, 20, 8, 17, 23, 25, 26]
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
    elif gmsh_cell == "tetra20":
        cell_shape = "tetrahedron"
        degree = 3
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

    mesh_2D(True, i)


def mesh_tetra(order):
    geom = pygmsh.opencascade.Geometry()
    geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
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
    mesh_tetra(i)
for i in ["1", "2"]:
    mesh_hex(i)
