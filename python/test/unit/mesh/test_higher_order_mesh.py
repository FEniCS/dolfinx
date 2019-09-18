# Copyright (C) 2019 Jørgen Schartum Dokken
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
""" Unit-tests for higher order meshes """

from dolfin import Mesh, MPI, Constant
from dolfin.cpp.mesh import CellType, GhostMode
from dolfin.fem import assemble_scalar
from ufl import dx, SpatialCoordinate, sin


import numpy as np
import pytest


@pytest.mark.parametrize('L', [1, 1, 2, 3])
@pytest.mark.parametrize('H', [0.5, 1, 2, 3])
def test_triangle_order_2(L, H):
    # Test second order mesh by computing volume two cells
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
