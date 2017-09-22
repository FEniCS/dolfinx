#!/usr/bin/env py.test

"""Unit tests for the Edge class"""

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
from dolfin_utils.test import fixture, skip_in_parallel


@fixture
def cube():
    return UnitCubeMesh(5, 5, 5)


@fixture
def square():
    return UnitSquareMesh(5, 5)


@pytest.fixture(scope='module', params=range(2))
def meshes(cube, square, request):
    mesh = [cube, square]
    return mesh[request.param]


@skip_in_parallel
def test_2DEdgeLength(square):
    """Iterate over edges and sum length."""
    length = 0.0
    for e in edges(square):
        length += e.length()
    assert round(length - 19.07106781186544708362, 7) == 0


@skip_in_parallel
def test_3DEdgeLength(cube):
    """Iterate over edges and sum length."""
    length = 0.0
    for e in edges(cube):
        length += e.length()
    assert round(length - 278.58049080280125053832, 7) == 0


def test_EdgeDot(meshes):
    """Iterate over edges compute dot product with ."""
    for e in edges(meshes):
        dot = e.dot(e)/(e.length()**2)
        assert round(dot - 1.0, 7) == 0
