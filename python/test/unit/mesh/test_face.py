# Copyright (C) 2011 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest

import dolfin
from dolfin import MPI, UnitCubeMesh, UnitSquareMesh
from dolfin_utils.test.fixtures import fixture
from dolfin_utils.test.skips import skip_in_parallel


@fixture
def cube():
    return UnitCubeMesh(MPI.comm_world, 5, 5, 5)


@fixture
def square():
    return UnitSquareMesh(MPI.comm_world, 5, 5)


@skip_in_parallel
def test_area(cube, square):
    """Iterate over faces and sum area."""
    cube.create_entities(2)
    area = dolfin.cpp.mesh.volume_entities(cube, range(cube.num_entities(2)), 2).sum()
    assert area == pytest.approx(39.21320343559672494393)

    cube.create_entities(1)
    area = dolfin.cpp.mesh.volume_entities(square, range(square.num_entities(2)), 2).sum()
    assert area == pytest.approx(1.0)
