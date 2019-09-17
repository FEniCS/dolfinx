# Copyright (C) 2019 Jørgen Schartum Dokken
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
""" Unit-tests for higher order meshes """

from dolfin import Mesh, MPI, Constant
from dolfin.cpp.mesh import CellType, GhostMode
# from dolfin.io import VTKFile
from dolfin.fem import assemble_scalar
from ufl import dx

import numpy as np
import pytest


def test_triangle_order_2():
    points = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]])
    cells = np.array([[0, 1, 2, 3, 4, 5]])
    # , [0,4,7,5,6,8]
    mesh = Mesh(MPI.comm_world, CellType.triangle, points, cells,
                [], GhostMode.none)

    vol = assemble_scalar(Constant(mesh, 1) * dx)
    assert(vol == pytest.approx(0.5, rel=1e-9))

    # VTKFile("mesh.pvd").write(mesh)


def test_triangle_order_3():
    H, L = 1, 2
    points = np.array([[0, 0], [L, 0], [0, H], [L / 3, 0], [2 * L / 3, 0],
                       [2 * L / 3, H / 3], [L / 3, 2 * H / 3], [0, 2 * H / 3],
                       [0, 2 * H / 3], [H / 3, H / 3]])
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    mesh = Mesh(MPI.comm_world, CellType.triangle, points, cells,
                [], GhostMode.none)
    # VTKFile("mesh.pvd").write(mesh)

    vol = assemble_scalar(Constant(mesh, 1) * dx)
    assert(vol == pytest.approx(H * L / 2, rel=1e-9))
