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
# Last changed: 2014-05-30

from __future__ import print_function
import pytest

from dolfin import *
from dolfin_utils.test import skip_in_parallel

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

@skip_in_parallel
def test_triangle_triangle_collision() :
    "Test that has been failing"
    assert collides_triangle_triangle_2d(Point(0.177432070718943, 0.5),
                                         Point(0.176638957524249, 0.509972290857582),
                                         Point(0.217189283468892, 0.550522616802225),
                                         Point(0.333333333333333, 0.52399308981973),
                                         Point(0.333333333333333, 0.666666666666667),
                                         Point(0.211774439087554, 0.545107772420888))


