#!/usr/bin/env py.test

"""Unit tests for intersection computation"""

# Copyright (C) 2013 Anders Logg
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
# First added:  2013-12-09
# Last changed: 2017-02-16

from __future__ import print_function
import pytest

from dolfin import *
from dolfin_utils.test import skip_in_parallel
import numpy as np

@skip_in_parallel
def test_issue_97():
    "Test from Mikael Mortensen (issue #97)"

    N = 2
    L = 1000
    mesh = BoxMesh(Point(0, 0, 0), Point(L, L, L), N, N, N)
    V = FunctionSpace(mesh, 'CG', 1)
    v = interpolate(Expression('x[0]', degree=1), V)
    x = Point(0.5*L, 0.5*L, 0.5*L)
    vx = v(x)

@skip_in_parallel
def test_issue_168():
    "Test from Torsten Wendav (issue #168)"

    mesh = UnitCubeMesh(14, 14, 14)
    V = FunctionSpace(mesh, "Lagrange", 1)
    v = Function(V)
    x = (0.75, 0.25, 0.125)
    vx = v(x)


@pytest.mark.skipif(True, reason="Since cell.contains(point) is doing an exact calculation, we cannot assume that the midpoint is exactly in the cell")
@skip_in_parallel
def test_segment_collides_point_3D_2():
    """Test case by Oyvind from https://bitbucket.org/fenics-project/dolfin/issue/296 for segment point collision in 3D"""
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, 1, 3)
    editor.init_vertices(2)
    editor.init_cells(1)
    editor.add_vertex(0, 41.06309891, 63.74219894, 68.10320282)
    editor.add_vertex(1, 41.45830154, 62.61560059, 66.43019867)
    editor.add_cell(0,0,1)
    editor.close()
    cell = Cell(mesh, 0)
    assert cell.contains(cell.midpoint())


def _test_collision_robustness_2d(aspect, y, step):
    nx = 10
    ny = int(aspect*nx)
    mesh = UnitSquareMesh(nx, ny, 'crossed')
    bb = mesh.bounding_box_tree()

    x = 0.0
    p = Point(x, y)
    while x <= 1.0:
        c = bb.compute_first_entity_collision(Point(x, y))
        assert c < np.uintc(-1)
        x += step

#@pytest.mark.skipif(True, reason="Not implemented in 3D")
@skip_in_parallel
def _test_collision_robustness_3d(aspect, y, z, step):
    nx = nz = 10
    ny = int(aspect*nx)
    mesh = UnitCubeMesh(nx, ny, nz)
    bb = mesh.bounding_box_tree()

    x = 0.0
    while x <= 1.0:
        c = bb.compute_first_entity_collision(Point(x, y, z))
        assert c < np.uintc(-1)
        x += step

@skip_in_parallel
@pytest.mark.slow
def test_collision_robustness_slow():
    """Test cases from https://bitbucket.org/fenics-project/dolfin/issue/296"""
    _test_collision_robustness_2d( 100, 1e-14,       1e-5)
    _test_collision_robustness_2d(  40, 1e-03,       1e-5)
    _test_collision_robustness_2d( 100, 0.5 + 1e-14, 1e-5)
    _test_collision_robustness_2d(4.43, 0.5,      4.03e-6)
    _test_collision_robustness_3d( 100, 1e-14, 1e-14, 1e-5)

@skip_in_parallel
@pytest.mark.slow
@pytest.mark.skipif(True, reason='Very slow test cases')
def test_collision_robustness_very_slow():
    """Test cases from https://bitbucket.org/fenics-project/dolfin/issue/296"""
    _test_collision_robustness_2d(  10, 1e-16,       1e-7)
    _test_collision_robustness_2d(4.43, 1e-17,    4.03e-6)
    _test_collision_robustness_2d(  40, 0.5,         1e-6)
    _test_collision_robustness_2d(  10, 0.5 + 1e-16, 1e-7)

@skip_in_parallel
def test_points_on_line():
    """Test case from https://bitbucket.org/fenics-project/dolfin/issues/790"""
    big = 1e6
    p1 = Point(0.1, 0.06)
    p3 = Point(big*2.1, big*0.1)
    p2 = Point(0.0, big*3.0)
    p0 = Point(big*3.0, 0.0)

    mesh = Mesh()
    ed = MeshEditor()
    ed.open(mesh, 2, 2)
    ed.init_cells(3)
    ed.init_vertices(4)
    ed.add_vertex(0, p0)
    ed.add_vertex(1, p1)
    ed.add_vertex(2, p2)
    ed.add_vertex(3, p3)

    ed.add_cell(0, 2, 3, 0)
    ed.add_cell(1, 0, 1, 3)
    ed.add_cell(2, 1, 2, 3)

    ed.close()

    # xdmf = XDMFFile("a.xdmf")
    # xdmf.write(mesh)

    # print mesh.cells()
    # print mesh.coordinates()

    bb = mesh.bounding_box_tree()

    # Find a point on line somewhere between p3 and p1
    j = 4
    pq = (p3*j + p1*(50-j))/50.0
    c = bb.compute_entity_collisions(pq)
    # print pq.str(), c

    # Check that the neighbouring points are in the correct cells
    cell_numbers = [1, 2, 2, 1, 1, 2, 1, 1, 2]
    step = 1e-6
    cnt = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            pt = pq + Point(step*i, step*j)
            c = bb.compute_entity_collisions(pt)
            assert c[0] == cell_numbers[cnt]
            cnt += 1
