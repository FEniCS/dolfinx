# Copyright (C) 2020 Garth N. Wells, Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx import cpp as _cpp
from dolfinx.cpp.mesh import cell_normals
from dolfinx.mesh import create_unit_cube, create_unit_square, locate_entities_boundary


@pytest.fixture
def cube():
    return create_unit_cube(MPI.COMM_WORLD, 5, 5, 5)


@pytest.fixture
def square():
    return create_unit_square(MPI.COMM_WORLD, 5, 5)


@pytest.mark.skip("volume_entities needs fixing")
@pytest.mark.skip_in_parallel
def test_area(cube, square):
    """Iterate over faces and sum area."""

    # TODO: update for dim < tdim
    # cube.topology.create_entities(2)
    # area = dolfinx.cpp.mesh.volume_entities(cube, range(cube.num_entities(2)), 2).sum()
    # assert area == pytest.approx(39.21320343559672494393)

    map = square.topology.index_map(2)
    num_faces = map.size_local + map.num_ghosts

    cube.topology.create_entities(1)
    area = _cpp.mesh.volume_entities(square, range(num_faces), 2).sum()
    assert area == pytest.approx(1.0)


def test_normals(cube, square):
    """Test cell normals for a subset of facets"""

    def left_side(x):
        return np.isclose(x[0], 0)

    fdim = cube.topology.dim - 1
    facets = locate_entities_boundary(cube, fdim, left_side)
    normals = cell_normals(cube._cpp_object, fdim, facets)
    assert np.allclose(np.abs(normals), [1, 0, 0])

    fdim = square.topology.dim - 1
    facets = locate_entities_boundary(square, fdim, left_side)
    normals = cell_normals(square._cpp_object, fdim, facets)
    assert np.allclose(np.abs(normals), [1, 0, 0])
