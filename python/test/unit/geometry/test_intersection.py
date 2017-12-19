"""Unit tests for intersection computation"""

# Copyright (C) 2013-2014 Anders Logg
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

import pytest

from dolfin import intersect
from dolfin import (UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh, BoxMesh, CellType)
from dolfin import Point, FunctionSpace, Expression, interpolate
from dolfin import MPI

from dolfin_utils.test import skip_in_parallel

@skip_in_parallel
def test_mesh_point_1d():
    "Test mesh-point intersection in 1D"

    point = Point(0.1)
    mesh = UnitIntervalMesh(16)

    intersection = intersect(mesh, point)

    assert intersection.intersected_cells() == [1]

@skip_in_parallel
def test_mesh_point_2d_triangle():
    "Test mesh-point intersection in 2D for triangular mesh"

    point = Point(0.1, 0.2)
    mesh = UnitSquareMesh(16, 16)

    intersection = intersect(mesh, point)

    assert intersection.intersected_cells() == [98]

@skip_in_parallel
def test_mesh_point_3d_tetrahedron():
    "Test mesh-point intersection in 3D for tetrahedral mesh"

    point = Point(0.1, 0.2, 0.3)
    mesh = UnitCubeMesh(8, 8, 8)

    intersection = intersect(mesh, point)

    assert intersection.intersected_cells() == [816]

@skip_in_parallel
@pytest.mark.xfail(strict=True, raises=RuntimeError)
def test_mesh_point_2d_quadrilateral():
    "Test mesh-point intersection in 2D for quadrilateral mesh"

    point = Point(0.1, 0.2)
    mesh = UnitSquareMesh.create(16, 16, CellType.Type.quadrilateral)

    intersection = intersect(mesh, point)

    assert intersection.intersected_cells() == [49]

@skip_in_parallel
@pytest.mark.xfail(strict=True, raises=RuntimeError)
def test_mesh_point_3d_hexahedron():
    "Test mesh-point intersection in 3D for hexahedral mesh"

    point = Point(0.1, 0.2, 0.3)
    mesh = UnitCubeMesh.create(8, 8, 8, CellType.Type.hexahedron)

    intersection = intersect(mesh, point)

    # Returns [] now, but [136] is the correct cell.
    assert intersection.intersected_cells() == [136]
