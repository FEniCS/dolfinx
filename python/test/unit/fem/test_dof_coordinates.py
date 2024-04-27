# Copyright (C) 2024 Matthew Scroggs and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx.fem import Function, functionspace
from dolfinx.mesh import create_unit_cube, create_unit_square


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("degree", range(1, 5))
def test_dof_coords_2d(degree, dtype):
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10, dtype=dtype)
    V = functionspace(mesh, ("Lagrange", degree))
    u = Function(V, dtype=dtype)
    u.interpolate(lambda x: x[0])
    u.x.scatter_forward()
    x = V.tabulate_dof_coordinates()
    np.allclose(u.x.array, x[:, 0], atol=1e-7, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("degree", range(1, 5))
def test_dof_coords_3d(degree, dtype):
    mesh = create_unit_cube(MPI.COMM_WORLD, 10, 10, 10, dtype=dtype)
    V = functionspace(mesh, ("Lagrange", degree))
    u = Function(V, dtype=dtype)
    u.interpolate(lambda x: x[0])
    u.x.scatter_forward()
    x = V.tabulate_dof_coordinates()

    eps = np.sqrt(np.finfo(dtype).eps)
    np.allclose(u.x.array, x[:, 0], atol=1e-7, rtol=eps)
