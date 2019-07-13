# Copyright (C) 2011 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
import pytest

from dolfin import MPI, Face, Faces, UnitCubeMesh, UnitSquareMesh
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
    area = 0.0
    cube.create_entities(2)
    for f in Faces(cube):
        area += f.area()
    assert round(area - 39.21320343559672494393, 7) == 0

    area = 0.0
    cube.create_entities(1)
    for f in Faces(square):
        area += f.area()
    assert round(area - 1.0, 7) == 0


@skip_in_parallel
def test_normal_point(cube, square):
    """Compute normal vector to each face."""
    cube.create_entities(2)
    for f in Faces(cube):
        n = f.normal()
        assert round(numpy.linalg.norm(n) - 1.0, 7) == 0

    f = Face(square, 0)
    with pytest.raises(RuntimeError):
        f.normal()


@skip_in_parallel
def test_normal_component(cube, square):
    """Compute normal vector components to each face."""
    cube.create_entities(2)
    for f in Faces(cube):
        n = f.normal()
        norm = sum([x * x for x in n])
        assert round(norm - 1.0, 7) == 0

    f = Face(square, 0)
    with pytest.raises(RuntimeError):
        f.normal()
