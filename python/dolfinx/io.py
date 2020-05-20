# Copyright (C) 2017-2020 Chris N. Richardson, Garth N. Wells and Michal Habera
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""IO module for input data, post-processing and checkpointing"""

import ufl
import numpy
from dolfinx import cpp, fem


class VTKFile:
    """Interface to VTK files
    VTK supports arbitrary order Lagrangian finite elements for the
    geometry description. XDMF is the preferred format for geometry
    order <= 2.

    """

    def __init__(self, filename: str):
        """Open VTK file
        Parameters
        ----------
        filename
            Name of the file
        """
        self._cpp_object = cpp.io.VTKFile(filename)

    def write(self, o, t=None) -> None:
        """Write object to file"""
        o_cpp = getattr(o, "_cpp_object", o)
        if t is None:
            self._cpp_object.write(o_cpp)
        else:
            self._cpp_object.write(o_cpp, t)


class XDMFFile(cpp.io.XDMFFile):
    def write_function(self, u, t=0.0, mesh_xpath="/Xdmf/Domain/Grid[@GridType='Uniform'][1]"):
        u_cpp = getattr(u, "_cpp_object", u)
        super().write_function(u_cpp, t, mesh_xpath)

    def read_mesh(self, ghost_mode=cpp.mesh.GhostMode.shared_facet, name="mesh", xpath="/Xdmf/Domain"):
        # Read mesh data from file
        cell_type, x, cells = super().read_mesh_data(name, xpath)

        # Construct the geometry map
        cell = ufl.Cell(cpp.mesh.to_string(cell_type[0]), geometric_dimension=x.shape[1])
        domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, cell_type[1]))
        cmap = fem.create_coordinate_map(domain)

        # Build the mesh
        mesh = cpp.mesh.create(self.comm(), cpp.graph.AdjacencyList64(cells), cmap, x, ghost_mode)
        mesh.name = name
        domain._ufl_cargo = mesh
        mesh._ufl_domain = domain

        return mesh


def gmsh_to_dolfin(gmsh_cell):
    """
    Returns the cell permutation of the local ordering on a dolfinx cell as the (G)MSH local node ordering.
    The MSH ordering is described at:
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


def dolfin_to_gmsh(gmsh_cell):
    """
    Returns the cell permutation of the local ordering on a (G)MSH cell as the local dolfin node ordering.
    The MSH ordering is described at:
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
    reverse = dolfin_to_gmsh(gmsh_cell)
    perm = numpy.zeros(len(reverse), dtype=numpy.int32)
    for i in range(len(reverse)):
        perm[reverse[i]] = i
    return perm


def ufl_mesh_from_gmsh(gmsh_cell, gdim):
    """
    return a UFL mesh given a gmsh cell string and the geometric dimension.
    """
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
