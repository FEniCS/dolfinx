# Copyright (C) 2020 Garth N. Wells, Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx.mesh import (
    cell_normals,
    create_unit_cube,
    create_unit_square,
    locate_entities_boundary,
)


@pytest.fixture
def cube():
    return create_unit_cube(MPI.COMM_WORLD, 5, 5, 5)


@pytest.fixture
def square():
    return create_unit_square(MPI.COMM_WORLD, 5, 5)


def test_normals(cube, square):
    """Test cell normals for a subset of facets."""

    def left_side(x):
        return np.isclose(x[0], 0)

    fdim = cube.topology.dim - 1
    facets = locate_entities_boundary(cube, fdim, left_side)
    normals = cell_normals(cube, fdim, facets)
    assert np.allclose(np.abs(normals), [1, 0, 0])

    fdim = square.topology.dim - 1
    facets = locate_entities_boundary(square, fdim, left_side)
    normals = cell_normals(square, fdim, facets)
    assert np.allclose(np.abs(normals), [1, 0, 0])
