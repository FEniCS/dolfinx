# Copyright (C) 2026 Jack S. Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx.fem import coordinate_element
from dolfinx.mesh import CellType, create_unit_square, interpolate_geometry


def _assert_close_up_to_row_permutation(a, b, atol):
    """Assert rows of a and b are equal, up to a row permutation"""
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape
    used = [False] * len(b)
    for ra in a:
        for j, rb in enumerate(b):
            if not used[j] and np.allclose(ra, rb, atol=atol, rtol=0.0):
                used[j] = True
                break
        else:
            raise AssertionError(f"No match for {ra} in {b}")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interpolate_geometry_p1_to_p2(dtype):
    msh = create_unit_square(MPI.COMM_WORLD, 4, 4, dtype=dtype)
    cmap = coordinate_element(CellType.triangle, 2, dtype=dtype)

    new_msh = interpolate_geometry(msh, cmap)
    assert new_msh.geometry.cmap().degree == 2

    # Topology should be the same cpp object.
    assert new_msh.topology._cpp_object is msh.topology._cpp_object

    x_old, x_new = msh.geometry.x, new_msh.geometry.x
    dm_old, dm_new = msh.geometry.dofmap, new_msh.geometry.dofmap
    assert dm_old.shape[0] == dm_new.shape[0]
    assert dm_new.shape[1] == 6

    atol = 10 * np.finfo(dtype).eps
    for c in range(dm_new.shape[0]):
        v = x_old[dm_old[c]]
        # P2 geometry has original vertices + midpoints.
        expected = np.vstack([v, 0.5 * (v[0] + v[1]), 0.5 * (v[0] + v[2]), 0.5 * (v[1] + v[2])])
        _assert_close_up_to_row_permutation(x_new[dm_new[c]], expected, atol=atol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interpolate_geometry_p1_roundtrip(dtype):
    msh = create_unit_square(MPI.COMM_WORLD, 4, 4, dtype=dtype)
    cmap = coordinate_element(CellType.triangle, 1, dtype=dtype)

    new_msh = interpolate_geometry(msh, cmap)

    assert new_msh.geometry.cmap().degree == 1
    assert new_msh.topology._cpp_object is msh.topology._cpp_object

    x_old, x_new = msh.geometry.x, new_msh.geometry.x
    dm_old, dm_new = msh.geometry.dofmap, new_msh.geometry.dofmap
    assert dm_old.shape == dm_new.shape

    atol = 10 * np.finfo(dtype).eps
    for c in range(dm_old.shape[0]):
        np.testing.assert_allclose(x_new[dm_new[c]], x_old[dm_old[c]], atol=atol, rtol=0.0)
