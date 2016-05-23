#!/usr/bin/env py.test

"""Unit tests for the CollisionDetection class"""

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
# Last changed: 2016-05-23

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
def eps():
    return np.finfo(np.float32).eps

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
    cell0 = Cell(mesh0, 0)
    cell1 = Cell(mesh1, 0)
    assert cell0.collides(cell1) == False

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
