# Copyright (C) 2019 Jørgen Schartum Dokken & Matthew Scroggs
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
""" Unit-tests for higher order meshes """

from dolfin import Mesh, MPI, Constant
from dolfin.cpp.mesh import CellType, GhostMode
from dolfin.fem import assemble_scalar
from dolfin_utils.test.skips import skip_in_parallel
from ufl import dx, SpatialCoordinate, sin


import numpy as np
import pytest


@skip_in_parallel
@pytest.mark.parametrize('L', [1, 1, 2, 3])
@pytest.mark.parametrize('H', [0.5, 1, 2, 3])
def test_triangle_order_2(L, H):
    # Test second order mesh by computing volume of two cells
    #  *-----*-----*   3----6-----2
    #  | \         |   | \        |
    #  |   \       |   |   \      |
    #  *     *     *   7     8    5
    #  |       \   |   |      \   |
    #  |         \ |   |        \ |
    #  *-----*-----*   0----4-----1

    # Perturbation of nodes 4,5,6,7 while keeping volume constant
    points = np.array([[0, 0], [L, 0], [L, H], [0, H],
                       [L / 2, 0], [L, H / 2], [L / 2, H],
                       [0, H / 2], [L / 2, H / 2]])
    cells = np.array([[0, 1, 3, 4, 8, 7], [1, 2, 3, 5, 6, 8]])
    mesh = Mesh(MPI.comm_world, CellType.triangle, points, cells,
                [], GhostMode.none)

    vol = assemble_scalar(Constant(mesh, 1) * dx)
    assert(vol == pytest.approx(L * H, rel=1e-9))

    # Volume of cell 1
    cell_1 = np.array([cells[0]])
    mesh = Mesh(MPI.comm_world, CellType.triangle, points, cell_1,
                [], GhostMode.none)
    vol = assemble_scalar(Constant(mesh, 1) * dx)
    assert(vol == pytest.approx(L * H / 2, rel=1e-9))

    # Volume of cell 2
    cell_2 = np.array([cells[1]])
    mesh = Mesh(MPI.comm_world, CellType.triangle, points, cell_2,
                [], GhostMode.none)
    vol = assemble_scalar(Constant(mesh, 1) * dx)

    assert(vol == pytest.approx(L * H / 2, rel=1e-9))


@skip_in_parallel
def test_triangle_order_3():
    H, L = 1, 1
    #  *---*---*---*   3--11--10--2
    #  | \         |   | \        |
    #  *   *   *   *   8   7  15  13
    #  |     \     |   |    \     |
    #  *  *    *   *   9  14  6   12
    #  |         \ |   |        \ |
    #  *---*---*---*   0--4---5---1
    points = np.array([[0, 0], [L, 0], [L, H], [0, H],          # 0, 1, 2, 3
                       [L / 3, 0], [2 * L / 3, 0],              # 4, 5
                       [2 * L / 3, H / 3], [L / 3, 2 * H / 3],  # 6, 7
                       [0, 2 * H / 3], [0, H / 3],              # 8, 9
                       [2 * L / 3, H], [L / 3, H],              # 10, 11
                       [L, H / 3], [L, 2 * H / 3],              # 12, 13
                       [H / 3, H / 3],                          # 14
                       [2 * L / 3, 2 * H / 3]])                 # 15

    cells = np.array([[0, 1, 3, 4, 5, 6, 7, 8, 9, 14],
                      [1, 2, 3, 12, 13, 10, 11, 7, 6, 15]])

    def quantities(mesh):
        x, y = SpatialCoordinate(mesh)
        q1 = assemble_scalar(x * y * dx)
        q2 = assemble_scalar(x * y * sin(x) * dx)
        q3 = assemble_scalar(Constant(mesh, 1) * dx)
        return q1, q2, q3

    # Only first cell as mesh
    cell_0 = np.array([cells[0]])
    mesh = Mesh(MPI.comm_world, CellType.triangle, points, cell_0,
                [], GhostMode.none)
    q1, q2, q3 = quantities(mesh)
    assert q1 == pytest.approx(1 / 24, rel=1e-9)
    assert q2 == pytest.approx(2 - 3 * np.sin(1) + np.cos(1), rel=1e-9)
    assert q3 == pytest.approx(L * H / 2)

    # Only second cell as mesh
    cell_0 = np.array([cells[1]])
    mesh = Mesh(MPI.comm_world, CellType.triangle, points, cell_0,
                [], GhostMode.none)
    q1, q2, q3 = quantities(mesh)
    assert q1 == pytest.approx(5 / 24, rel=1e-9)
    assert q2 == pytest.approx(0.5 * (-4 + 7 * np.sin(1) - 3 * np.cos(1)),
                               rel=1e-9)
    assert q3 == pytest.approx(L * H / 2)

    # Both cells as mesh
    mesh = Mesh(MPI.comm_world, CellType.triangle, points, cells,
                [], GhostMode.none)
    q1, q2, q3 = quantities(mesh)
    assert q1 == pytest.approx(0.25, rel=1e-9)
    assert q2 == pytest.approx(0.5 * (np.sin(1) - np.cos(1)),
                               rel=1e-9)
    assert q3 == pytest.approx(L * H)


@skip_in_parallel
@pytest.mark.parametrize('L', [1, 4])
@pytest.mark.parametrize('H', [1, 5])
@pytest.mark.parametrize('eps', [0, 1, 10, 100])
def test_quad_dofs_order_2(L, H, eps):
    # Test second order mesh by computing volume of two cells
    #  *-----*-----*   3--6--2--13-10
    #  |     |     |   |     |     |
    #  |     |     |   7  8  5  14 12
    #  |     |     |   |     |     |
    #  *-----*-----*   0--4--1--11-9
    points = np.array([[0, 0], [L, 0], [L, H], [0, H],
                       [L / 2, 0], [L + eps, H / 2], [L / 2, H], [0 + eps, H / 2],
                       [L / 2 + eps, H / 2],
                       [2 * L, 0], [2 * L, H],
                       [3 * L / 2, 0], [2 * L + eps, H / 2], [3 * L / 2, H],
                       [3 * L / 2 + eps, H / 2]])
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 9, 10, 2, 11, 12, 13, 5, 14]])

    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points, cells,
                [], GhostMode.none)

    vol = assemble_scalar(Constant(mesh, 1) * dx)
    assert(vol == pytest.approx(L * H * 2, rel=1e-9))

    # Volume of cell 1
    cell_1 = np.array([cells[0]])
    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points, cell_1,
                [], GhostMode.none)
    vol = assemble_scalar(Constant(mesh, 1) * dx)
    assert(vol == pytest.approx(L * H, rel=1e-9))

    # Volume of cell 2
    cell_2 = np.array([cells[1]])
    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points, cell_2,
                [], GhostMode.none)
    vol = assemble_scalar(Constant(mesh, 1) * dx)
    assert(vol == pytest.approx(L * H, rel=1e-9))


@skip_in_parallel
@pytest.mark.parametrize('L', [1, 4])
@pytest.mark.parametrize('H', [1, 5])
@pytest.mark.parametrize('eps', [0, 1, 10, 100])
def test_quad_dofs_order_3(L, H, eps):
    # Test third order mesh by computing volume of two cells
    #  *---------*---------*   3--9--8--2-23-22-17
    #  |         |         |   |        |       |
    #  |         |         |   10 15 14 7 27 26 21
    #  |         |         |   |        |       |
    #  |         |         |   11 12 13 6 24 25 20
    #  |         |         |   |        |       |
    #  *---------*---------*   0--4--5--1-18-19-16
    points = np.array([[0, 0], [L, 0], [L, H], [0, H],
                       #
                       [L / 3, 0], [2 * L / 3, 0],
                       [L + eps, H / 3], [L - eps, 2 * H / 3],
                       [2 * L / 3, H], [L / 3, H],
                       [-eps, 2 * H / 3], [eps, H / 3],
                       #
                       [L / 3 + eps, H / 3], [2 * L / 3 + eps, H / 3],
                       [2 * L / 3 - eps, 2 * H / 3], [L / 3 - eps, 2 * H / 3],
                       #####
                       [2 * L, 0], [2 * L, H],
                       #
                       [4 * L / 3, 0], [5 * L / 3, 0],
                       [2 * L + eps, H / 3], [2 * L - eps, 2 * H / 3],
                       [5 * L / 3, H], [4 * L / 3, H],
                       #
                       [4 * L / 3 + eps, H / 3], [5 * L / 3 + eps, H / 3],
                       [5 * L / 3 - eps, 2 * H / 3], [4 * L / 3 - eps, 2 * H / 3]])
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                      [1, 16, 17, 2, 18, 19, 20, 21, 22, 23, 7, 6, 24, 25, 26, 27]])

    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points, cells,
                [], GhostMode.none)

    vol = assemble_scalar(Constant(mesh, 1) * dx)
    assert(vol == pytest.approx(L * H * 2, rel=1e-9))

    # Volume of cell 1
    cell_1 = np.array([cells[0]])
    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points, cell_1,
                [], GhostMode.none)
    vol = assemble_scalar(Constant(mesh, 1) * dx)
    assert(vol == pytest.approx(L * H, rel=1e-9))

    # Volume of cell 2
    cell_2 = np.array([cells[1]])
    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points, cell_2,
                [], GhostMode.none)
    vol = assemble_scalar(Constant(mesh, 1) * dx)
    assert(vol == pytest.approx(L * H, rel=1e-9))


@skip_in_parallel
@pytest.mark.parametrize('L', [1, 4])
@pytest.mark.parametrize('H', [1, 5])
@pytest.mark.parametrize('eps', [0, 1, 10, 100])
def test_quad_dofs_order_4(L, H, eps):
    # Test third order mesh by computing volume of one cell
    #  *---------*   3--10-11-12-2
    #  |         |   |           |
    #  |         |   15 19 22 18 9
    #  |         |   |           |
    #  |         |   14 23 24 21 8
    #  |         |   |           |
    #  |         |   13 16 20 17 7
    #  |         |   |           |
    #  *---------*   0--4--5--6--1
    points = np.array([[0, 0], [L, 0], [L, H], [0, H],
                       #
                       [L / 4, 0], [L / 2, 0], [3 * L / 4, 0],
                       [L + eps, H / 4], [L - eps, H / 2], [L + eps, 3 * H / 4],
                       [L / 4, H], [L / 2, H], [3 * L / 4, H],
                       [eps, H / 4], [-eps, H / 2], [eps, 3 * H / 4],
                       #
                       [L / 4, H / 4], [L / 2, H / 4], [3 * L / 4, H / 4],
                       [L / 4, H / 2], [L / 2, H / 2], [3 * L / 4, H / 2],
                       [L / 4, 3 * H / 4], [L / 2, 3 * H / 4], [3 * L / 4, 3 * H / 4]])
    cells = np.array([list(range(25))])

    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points, cells,
                [], GhostMode.none)

    vol = assemble_scalar(Constant(mesh, 1) * dx)
    assert(vol == pytest.approx(L * H, rel=1e-9))

    # Volume of cell 1
    cell_1 = np.array([cells[0]])
    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points, cell_1,
                [], GhostMode.none)
    vol = assemble_scalar(Constant(mesh, 1) * dx)
    assert(vol == pytest.approx(L * H, rel=1e-9))
