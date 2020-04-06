# Copyright (C) 2011 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest

import dolfinx
from dolfinx import MPI, UnitCubeMesh, UnitSquareMesh
from dolfinx_utils.test.skips import skip_in_parallel


@pytest.fixture
def cube():
    return UnitCubeMesh(MPI.comm_world, 5, 5, 5)


@pytest.fixture
def square():
    return UnitSquareMesh(MPI.comm_world, 5, 5)


@skip_in_parallel
def test_area(cube, square):
    """Iterate over faces and sum area."""

    # TODO: update for dim < tdim
    # cube.create_entities(2)
    # area = dolfinx.cpp.mesh.volume_entities(cube, range(cube.num_entities(2)), 2).sum()
    # assert area == pytest.approx(39.21320343559672494393)

    map = square.topology.index_map(2)
    num_faces = map.size_local + map.num_ghosts

    cube.create_entities(1)
    area = dolfinx.cpp.mesh.volume_entities(square, range(num_faces), 2).sum()
    assert area == pytest.approx(1.0)
