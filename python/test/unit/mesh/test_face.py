# Copyright (C) 2020 Garth N. Wells, Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
import pytest
import numpy
from dolfinx import UnitCubeMesh, UnitSquareMesh
from dolfinx.mesh import locate_entities_boundary
from dolfinx.cpp.mesh import cell_normals
from dolfinx_utils.test.skips import skip_in_parallel
from mpi4py import MPI


@pytest.fixture
def cube():
    return UnitCubeMesh(MPI.COMM_WORLD, 5, 5, 5)


@pytest.fixture
def square():
    return UnitSquareMesh(MPI.COMM_WORLD, 5, 5)


@skip_in_parallel
def test_area(cube, square):
    """Iterate over faces and sum area."""

    # TODO: update for dim < tdim
    # cube.topology.create_entities(2)
    # area = dolfinx.cpp.mesh.volume_entities(cube, range(cube.num_entities(2)), 2).sum()
    # assert area == pytest.approx(39.21320343559672494393)

    map = square.topology.index_map(2)
    num_faces = map.size_local + map.num_ghosts

    cube.topology.create_entities(1)
    area = dolfinx.cpp.mesh.volume_entities(square, range(num_faces), 2).sum()
    assert area == pytest.approx(1.0)


def test_normals(cube, square):
    """ Test that cell normals of a set of facets """
    def left_side(x):
        return numpy.isclose(x[0], 0)
    fdim = cube.topology.dim - 1
    facets = locate_entities_boundary(cube, fdim, left_side)
    normals = cell_normals(cube, fdim, facets)
    assert(numpy.isclose(normals, [-1, 0, 0]).all(axis=1).all())

    fdim = square.topology.dim - 1
    facets = locate_entities_boundary(square, fdim, left_side)
    normals = cell_normals(square, fdim, facets)
    assert(numpy.isclose(normals, [-1, 0, 0]).all(axis=1).all())
