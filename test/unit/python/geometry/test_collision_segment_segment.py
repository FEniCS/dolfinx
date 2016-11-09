#!/usr/bin/env py.test

"""Unit tests for the CollisionPredicates class"""

# Copyright (C) 2014 Anders Logg and August Johansson
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
# First added:  2014-02-16
# Last changed: 2016-11-09

import pytest
from dolfin import *
from dolfin_utils.test import skip_in_parallel
import numpy as np

@skip_in_parallel
def create_mesh(a, b):
    editor = MeshEditor()
    mesh = Mesh()
    editor.open(mesh,1,2)
    editor.init_cells(1)
    editor.init_vertices(2)
    editor.add_cell(0, 0, 1)
    editor.add_vertex(0, a)
    editor.add_vertex(1, b)
    editor.close()
    return mesh;

@skip_in_parallel
def test_L_version_1():
    mesh0 = create_mesh(Point(0., 0.), Point(1., 0.))
    mesh1 = create_mesh(Point(0., 0.), Point(0., 1.))
    cell0 = Cell(mesh0, 0)
    cell1 = Cell(mesh1, 0)
    assert cell0.collides(cell1) == True

@skip_in_parallel
def test_L_version_2():
    # mesh0 = create_mesh(Point(np.finfo(np.float32).eps, 0.), Point(1., 0.))
    # mesh0 = create_mesh(Point(eps(), 0.), Point(1., 0.))
    mesh0 = create_mesh(Point(2.23e-15, 0.), Point(1., 0.))
    mesh1 = create_mesh(Point(0., 0.), Point(0., 1.))
    print(mesh0.str(True))
    print(mesh1.str(True))
    cell0 = Cell(mesh0, 0)
    cell1 = Cell(mesh1, 0)
    assert cell0.collides(cell1) == False

@skip_in_parallel
def test_L_version_3():
    # mesh0 = create_mesh(Point(np.finfo(np.float32).eps, 0.), Point(1., 0.))
    # mesh0 = create_mesh(Point(eps(), 0.), Point(1., 0.))
    a = Point(2.23e-100, 0.) # assume shewchuk works
    b = Point(1., 0.)
    c = Point(0., 0.)
    d = Point(0., 1.)

    assert CollisionPredicates.collides_segment_segment_2d(a, b, c, d) == False

@skip_in_parallel
def test_aligned_version_1():
    mesh0 = create_mesh(Point(0,0), Point(1,0))
    mesh1 = create_mesh(Point(1,0), Point(2,0))
    cell0 = Cell(mesh0, 0)
    cell1 = Cell(mesh1, 0)
    assert cell0.collides(cell1) == True

@skip_in_parallel
def test_aligned_version_2():
    mesh0 = create_mesh(Point(0,0), Point(1,0))
    mesh1 = create_mesh(Point(2,0), Point(3,0))
    cell0 = Cell(mesh0, 0)
    cell1 = Cell(mesh1, 0)
    assert cell0.collides(cell1) == False

@skip_in_parallel
def test_parallel_1():
    p0 = Point(0.05, 0.15)
    p1 = Point(0.85, 0.95)
    q0 = Point(0.2875, 0.3875)
    q1 = Point(0.6125, 0.7125)
    meshp = create_mesh(p0, p1)
    meshq = create_mesh(q0, q1)
    cellp = Cell(meshp, 0)
    cellq = Cell(meshq, 0)
    assert cellp.collides(cellq) == True

@skip_in_parallel
def test_parallel_2():
    res = CollisionPredicates.collides_segment_segment_2d(Point(.5, .3),
                                                          Point(.5, .4),
                                                          Point(.5, .5),
                                                          Point(.5, .6))
    assert not res

@skip_in_parallel
def test_point_on_segment():
    # p0 is on segment q0-q1
    p0 = Point(1e-17, 0)
    p1 = Point(1, 2)
    p2 = Point(2, 1)
    q0 = Point(1, 0)
    q1 = Point(0, 0)
    assert CollisionPredicates.collides_segment_segment_2d(p0, p1, q0, q1)
    assert CollisionPredicates.collides_segment_segment_2d(p0, p2, q0, q1)
    assert CollisionPredicates.collides_segment_segment_2d(p0, p1, q1, q0)
    assert CollisionPredicates.collides_segment_segment_2d(p0, p2, q1, q0)
