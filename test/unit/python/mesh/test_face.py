#!/usr/bin/env py.test

"""Unit tests for the Face class"""

# Copyright (C) 2011 Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2011-02-26
# Last changed: 2014-05-30

import pytest
from dolfin import *
from dolfin_utils.test import skip_in_parallel, fixture


@fixture
def cube():
    return UnitCubeMesh(5, 5, 5)


@fixture
def square():
    return UnitSquareMesh(5, 5)


@skip_in_parallel
def test_Area(cube, square):
    """Iterate over faces and sum area."""

    area = 0.0
    for f in faces(cube):
        area += f.area()
    assert round(area - 39.21320343559672494393, 7) == 0

    area = 0.0
    for f in faces(square):
        area += f.area()
    assert round(area - 1.0, 7) == 0


@skip_in_parallel
def test_NormalPoint(cube, square):
    """Compute normal vector to each face."""
    for f in faces(cube):
        n = f.normal()
        assert round(n.norm() - 1.0, 7) == 0

    f = Face(square, 0)
    with pytest.raises(RuntimeError):
        f.normal()


@skip_in_parallel
def test_NormalComponent(cube, square):
    """Compute normal vector components to each face."""
    D = cube.topology().dim()
    for f in faces(cube):
        n = [f.normal(i) for i in range(D)]
        norm = sum([x*x for x in n])
        assert round(norm - 1.0, 7) == 0

    f = Face(square, 0)
    with pytest.raises(RuntimeError):
        f.normal(0)
