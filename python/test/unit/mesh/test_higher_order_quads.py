# Copyright (C) 2019 Jørgen Schartum Dokken & Matthew Scroggs
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
""" Unit-tests for higher order quadrilateral meshes """

from dolfin import Mesh, MPI, fem, Function, FunctionSpace
from dolfin.cpp.mesh import CellType, GhostMode
from dolfin.fem import assemble_scalar
from dolfin_utils.test.skips import skip_in_parallel
from dolfin.cpp.io import vtk_to_dolfin_ordering
from ufl import dx

import numpy as np
import pygmsh
import pytest
from test_higher_order_triangles import sympy_scipy

@skip_in_parallel
@pytest.mark.parametrize('L', [1, 2])
@pytest.mark.parametrize('H', [1])
@pytest.mark.parametrize('Z', [0, 0.3])
def test_second_order_mesh(L, H, Z):
    # Test by comparing integration of z+x*y against sympy/scipy integration
    # of a quad element. Z>0 implies curved element.
    #  *-----*   3--6--2
    #  |     |   |     |
    #  |     |   7  8  5
    #  |     |   |     |
    #  *-----*   0--4--1
    points = np.array([[0, 0, 0], [L, 0, 0], [L, H, Z], [0, H, Z],
                       [L / 2, 0, 0], [L, H / 2, 0],
                       [L / 2, H, Z], [0, H / 2, 0],
                       [L / 2, H / 2, 0],
                       [2 * L, 0, 0], [2 * L, H, Z]])
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]])
    cells = vtk_to_dolfin_ordering(cells, CellType.quadrilateral)

    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points, cells,
                [], GhostMode.none)

    def e2(x):
        values = np.empty((x.shape[0], 1))
        values[:, 0] = x[:, 2] + x[:, 0] * x[:, 1]
        return values

    # Interpolate function
    V = FunctionSpace(mesh, ("CG", 2))
    u = Function(V)
    cmap = fem.create_coordinate_map(mesh.ufl_domain())

    mesh.geometry.coord_mapping = cmap

    u.interpolate(e2)

    intu = assemble_scalar(u * dx(mesh))
    intu = MPI.sum(mesh.mpi_comm(), intu)

    nodes = [0, 3, 7]
    ref = sympy_scipy(points, nodes, L, H)
    assert ref == pytest.approx(intu, rel=1e-6)


@skip_in_parallel
@pytest.mark.parametrize('L', [1, 2])
@pytest.mark.parametrize('H', [1])
@pytest.mark.parametrize('Z', [0, 0.3])
def test_third_order_mesh(L, H, Z):
    # Test by comparing integration of z+x*y against sympy/scipy integration
    # of a quad element. Z>0 implies curved element.
    #  *---------*   3--8--9--2-22-23-17
    #  |         |   |        |       |
    #  |         |   11 14 15 7 26 27 21
    #  |         |   |        |       |
    #  |         |   10 12 13 6 24 25 20
    #  |         |   |        |       |
    #  *---------*   0--4--5--1-18-19-16
    points = np.array([[0, 0, 0], [L, 0, 0], [L, H, Z], [0, H, Z],        # 0  1 2 3
                       [L / 3, 0, 0], [2 * L / 3, 0, 0],                  # 4  5
                       [L, H / 3, 0], [L, 2 * H / 3, 0],                  # 6  7
                       [L / 3, H, Z], [2 * L / 3, H, Z],                  # 8  9
                       [0, H / 3, 0], [0, 2 * H / 3, 0],                  # 10 11
                       [L / 3, H / 3, 0], [2 * L / 3, H / 3, 0],          # 12 13
                       [L / 3, 2 * H / 3, 0], [2 * L / 3, 2 * H / 3, 0],  # 14 15
                       [2 * L, 0, 0], [2 * L, H, 0],                      # 16 17
                       [4 * L / 3, 0, 0], [5 * L / 3, 0, 0],              # 18 19
                       [2 * L, H / 3, 0], [2 * L, 2 * H / 3, 0],          # 20 21
                       [4 * L / 3, H, 0], [5 * L / 3, H, 0],              # 22 23
                       [4 * L / 3, H / 3, 0], [5 * L / 3, H / 3, 0],           # 24 25
                       [4 * L / 3, 2 * H / 3, 0], [5 * L / 3, 2 * H / 3, 0]])  # 26 27

    # Change to multiple cells when matthews dof-maps work for quads
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
    # ,[1, 16, 17, 2, 18, 19, 20, 21, 22, 23, 6, 7, 24, 25, 26, 27]])

    cells = vtk_to_dolfin_ordering(cells, CellType.quadrilateral)
    print(cells)
    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points, cells,
                [], GhostMode.none)

    def e2(x):
        values = np.empty((x.shape[0], 1))
        values[:, 0] = x[:, 2] + x[:, 0] * x[:, 1]
        return values

    # Interpolate function
    V = FunctionSpace(mesh, ("CG", 3))
    u = Function(V)
    cmap = fem.create_coordinate_map(mesh.ufl_domain())

    mesh.geometry.coord_mapping = cmap

    u.interpolate(e2)

    intu = assemble_scalar(u * dx(mesh))
    intu = MPI.sum(mesh.mpi_comm(), intu)

    nodes = [0, 3, 10, 11]
    ref = sympy_scipy(points, nodes, L, H)
    assert ref == pytest.approx(intu, rel=1e-6)


@skip_in_parallel
@pytest.mark.parametrize('L', [1, 2])
@pytest.mark.parametrize('H', [1])
@pytest.mark.parametrize('Z', [0, 0.3])
def test_fourth_order_mesh(L, H, Z):
    # Test by comparing integration of z+x*y against sympy/scipy integration
    # of a quad element. Z>0 implies curved element.

    #  *---------*   20-21-22-23-24-41--42--43--44
    #  |         |   |           |              |
    #  |         |   15 16 17 18 19 37  38  39  40
    #  |         |   |           |              |
    #  |         |   10 11 12 13 14 33  34  35  36
    #  |         |   |           |              |
    #  |         |   5  6  7  8  9  29  30  31  32
    #  |         |   |           |              |
    #  *---------*   0--1--2--3--4--25--26--27--28
    points = np.array([[0, 0, 0], [L / 4, 0, 0], [L / 2, 0, 0],               # 0 1 2
                       [3 * L / 4, 0, 0], [L, 0, 0],                          # 3 4
                       [0, H / 4, -Z / 3], [L / 4, H / 4, -Z / 3], [L / 2, H / 4, -Z / 3],   # 5 6 7
                       [3 * L / 4, H / 4, -Z / 3], [L, H / 4, -Z / 3],                  # 8 9
                       [0, H / 2, 0], [L / 4, H / 2, 0], [L / 2, H / 2, 0],   # 10 11 12
                       [3 * L / 4, H / 2, 0], [L, H / 2, 0],                  # 13 14
                       [0, (3 / 4) * H, 0], [L / 4, (3 / 4) * H, 0],          # 15 16
                       [L / 2, (3 / 4) * H, 0], [3 * L / 4, (3 / 4) * H, 0],  # 17 18
                       [L, (3 / 4) * H, 0], [0, H, Z], [L / 4, H, Z],         # 19 20 21
                       [L / 2, H, Z], [3 * L / 4, H, Z], [L, H, Z],           # 22 23 24
                       [(5 / 4) * L, 0, 0], [(6 / 4) * L, 0, 0],              # 25 26
                       [(7 / 4) * L, 0, 0], [2 * L, 0, 0],                    # 27 28
                       [(5 / 4) * L, H / 4, -Z / 3], [(6 / 4) * L, H / 4, -Z / 3],      # 29 30
                       [(7 / 4) * L, H / 4, -Z / 3], [2 * L, H / 4, -Z / 3],            # 31 32
                       [(5 / 4) * L, H / 2, 0], [(6 / 4) * L, H / 2, 0],      # 33 34
                       [(7 / 4) * L, H / 2, 0], [2 * L, H / 2, 0],            # 35 36
                       [(5 / 4) * L, 3 / 4 * H, 0],                           # 37
                       [(6 / 4) * L, 3 / 4 * H, 0],                           # 38
                       [(7 / 4) * L, 3 / 4 * H, 0], [2 * L, 3 / 4 * H, 0],    # 39 40
                       [(5 / 4) * L, H, Z], [(6 / 4) * L, H, Z],              # 41 42
                       [(7 / 4) * L, H, Z], [2 * L, H, Z]])                   # 43 44

    # VTK ordering
    cells = np.array([[0, 4, 24, 20, 1, 2, 3, 9, 14, 19, 21, 22, 23, 5, 10, 15, 6, 7, 8, 11, 12, 13, 16, 17, 18]])
    #  , [4, 28, 44, 24, 25, 26, 27, 32, 36, 40, 41, 42, 43, 9, 14, 19,
    #     29, 30, 31, 33, 34, 35, 37, 38, 39]])

    cells = vtk_to_dolfin_ordering(cells, CellType.quadrilateral)

    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points, cells,
                [], GhostMode.none)

    def e2(x):
        values = np.empty((x.shape[0], 1))
        values[:, 0] = x[:, 2] + x[:, 0] * x[:, 1]
        return values

    V = FunctionSpace(mesh, ("CG", 4))
    u = Function(V)
    cmap = fem.create_coordinate_map(mesh.ufl_domain())

    mesh.geometry.coord_mapping = cmap

    u.interpolate(e2)

    intu = assemble_scalar(u * dx(mesh))
    intu = MPI.sum(mesh.mpi_comm(), intu)

    nodes = [0, 5, 10, 15, 20]
    ref = sympy_scipy(points, nodes, L, H)
    assert ref == pytest.approx(intu, rel=1e-5)

@skip_in_parallel
@pytest.mark.parametrize('order', [2, 3])
def test_gmsh_input(order):
    # Parameterize test if gmsh gets wider support
    R = 1
    res = 0.2 if order == 2 else 0.2
    algorithm = 2 if order == 2 else 5
    element = "quad{0:d}".format(int((order + 1)**2))

    geo = pygmsh.opencascade.Geometry()
    geo.add_raw_code("Mesh.ElementOrder={0:d};".format(order))
    geo.add_ball([0, 0, 0], R, char_length=res)
    geo.add_raw_code("Recombine Surface {1};")
    geo.add_raw_code("Mesh.Algorithm = {0:d};".format(algorithm))

    msh = pygmsh.generate_mesh(geo, verbose=True, dim=2)

    if order > 2:
        # Quads order > 3 have a gmsh specific ordering, and has to be permuted.
        msh_to_dolfin = np.array([0, 3, 11, 10, 1, 2, 6, 7, 4, 9, 12, 15, 5, 8, 13, 14])
        cells = np.zeros(msh.cells[element].shape)
        for i in range(len(cells)):
            for j in range(len(msh_to_dolfin)):
                cells[i, j] = msh.cells[element][i, msh_to_dolfin[j]]
    else:
        # XDMF does not support higher order quads
        cells = vtk_to_dolfin_ordering(msh.cells[element], CellType.quadrilateral)

    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, msh.points, cells,
                [], GhostMode.none)
    surface = assemble_scalar(1 * dx(mesh))

    assert MPI.sum(mesh.mpi_comm(), surface) == pytest.approx(4 * np.pi * R * R, rel=1e-5)

    # Bug related to VTK output writing
    # def e2(x):
    #     values = np.empty((x.shape[0], 1))
    #     values[:, 0] = x[:, 0]
    #     return values
    # cmap = fem.create_coordinate_map(mesh.ufl_domain())
    # mesh.geometry.coord_mapping = cmap
    # V = FunctionSpace(mesh, ("CG", order))
    # u = Function(V)
    # u.interpolate(e2)
    # from dolfin.io import VTKFile
    # VTKFile("u{0:d}.pvd".format(order)).write(u)
    # print(min(u.vector.array),max(u.vector.array))
    # print(assemble_scalar(u*dx(mesh)))
