# Copyright (C) 2013 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
import numpy

from dolfin import (UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh,
                    MPI, Cell, Mesh, CellType, Point)
from dolfin_utils.test import skip_in_parallel, skip_in_release


@skip_in_parallel
def test_distance_interval():

    mesh = UnitIntervalMesh(MPI.comm_self, 1)
    cell = Cell(mesh, 0)

    assert round(cell.distance(Point(-1.0)) - 1.0, 7) == 0
    assert round(cell.distance(Point(0.5)) - 0.0, 7) == 0


@skip_in_parallel
def test_distance_triangle():

    mesh = UnitSquareMesh(MPI.comm_self, 1, 1)
    cell = Cell(mesh, 1)

    assert round(cell.distance(Point(-1.0, -1.0)) - numpy.sqrt(2), 7) == 0
    assert round(cell.distance(Point(-1.0, 0.5)) - 1, 7) == 0
    assert round(cell.distance(Point(0.5, 0.5)) - 0.0, 7) == 0


@skip_in_parallel
def test_distance_tetrahedron():

    mesh = UnitCubeMesh(MPI.comm_self, 1, 1, 1)
    cell = Cell(mesh, 5)

    assert round(cell.distance(Point(-1.0, -1.0, -1.0))-numpy.sqrt(3), 7) == 0
    assert round(cell.distance(Point(-1.0, 0.5, 0.5)) - 1, 7) == 0
    assert round(cell.distance(Point(0.5, 0.5, 0.5)) - 0.0, 7) == 0


@pytest.mark.xfail
@skip_in_release
@skip_in_parallel
def test_issue_568():
    mesh = UnitSquareMesh(MPI.comm_self, 4, 4)
    cell = Cell(mesh, 0)

    # This no longer fails because serial mesh now is building facets, using
    # same pipeline as parallel.

    # Should throw an error, not just segfault (only works in DEBUG mode!)
    with pytest.raises(RuntimeError):
        cell.facet_area(0)

    # Should work after initializing the connectivity
    mesh.init(2, 1)
    cell.facet_area(0)


def test_volume_quadrilateralR2():

    mesh = UnitSquareMesh(MPI.comm_self, 1, 1, CellType.Type.quadrilateral)
    cell = Cell(mesh, 0)

    assert cell.volume() == 1.0


@pytest.mark.parametrize('coordinates', [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]])
def test_volume_quadrilateralR3(coordinates):

    mesh = Mesh(MPI.comm_world, CellType.Type.quadrilateral,
                numpy.array(coordinates, dtype=numpy.float64),
                numpy.array([[0,1,2,3]], dtype=numpy.int32))

    mesh.init()
    cell = Cell(mesh, 0)

    assert cell.volume() == 1.0


@pytest.mark.parametrize('scaling', [1e0, 1e-5, 1e-10, 1e-15, 1e-20, 1e-30,
    1e5, 1e10, 1e15, 1e20, 1e30])
def test_volume_quadrilateral_coplanarity_check_1(scaling):

    with pytest.raises(RuntimeError) as error:
        # Unit square cell scaled down by 'scaling' and the first
        # vertex is distorted so that the vertices are clearly non
        # coplanar
        mesh = Mesh(MPI.comm_world, CellType.Type.quadrilateral,
                    numpy.array([[scaling, 0.5 * scaling, 0.6 *
                                  scaling], [0.0, scaling, 0.0], [0.0, 0.0,
                                                                  scaling], [0.0, scaling, scaling]],
                                dtype=numpy.float64), numpy.array([[0, 1, 2, 3]],
                                                                  dtype=numpy.int32))

        mesh.init()
        cell = Cell(mesh, 0)
        volume = cell.volume()

    assert "are not coplanar" in str(error.value)


# Test when |p0-p3| is ~ 1 but |p1-p2| is small
# The cell is degenerate when scale is below 1e-17, it is expected to fail the test.
@pytest.mark.parametrize('scaling', [1e0, 1e-5, 1e-10, 1e-15])
def test_volume_quadrilateral_coplanarity_check_2(scaling):

    with pytest.raises(RuntimeError) as error:
        # Unit square cell scaled down by 'scaling' and the first
        # vertex is distorted so that the vertices are clearly non
        # coplanar
        mesh = Mesh(MPI.comm_world, CellType.Type.quadrilateral,
                    numpy.array([[1.0, 0.5, 0.6], [0.0, scaling, 0.0],
                                 [0.0, 0.0, scaling], [0.0, 1.0, 1.0]],
                                dtype=numpy.float64), numpy.array([[0, 1, 2, 3]],
                                                                  dtype=numpy.int32))
        mesh.init()
        cell = Cell(mesh, 0)
        volume = cell.volume()

    assert "are not coplanar" in str(error.value)
