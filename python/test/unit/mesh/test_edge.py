# Copyright (C) 2011 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
from dolfin import UnitSquareMesh, UnitCubeMesh, MPI, Edge, Edges
from dolfin_utils.test import fixture, skip_in_parallel


@fixture
def cube():
    return UnitCubeMesh(MPI.comm_world, 5, 5, 5)


@fixture
def square():
    return UnitSquareMesh(MPI.comm_world, 5, 5)


@pytest.fixture(scope='module', params=range(2))
def meshes(cube, square, request):
    mesh = [cube, square]
    return mesh[request.param]


@skip_in_parallel
def test_2DEdgeLength(square):
    """Iterate over edges and sum length."""
    length = 0.0
    square.init(1)
    print(square.num_entities(1))
    for e in Edges(square):
        length += e.length()
    assert round(length - 19.07106781186544708362, 7) == 0


@skip_in_parallel
def test_3DEdgeLength(cube):
    """Iterate over edges and sum length."""
    length = 0.0
    cube.init(1)
    for e in Edges(cube):
        length += e.length()
    assert round(length - 278.58049080280125053832, 7) == 0


def test_EdgeDot(meshes):
    """Iterate over edges compute dot product with ."""
    meshes.init(1)
    for e in Edges(meshes):
        dot = e.dot(e) / (e.length()**2)
        assert round(dot - 1.0, 7) == 0
