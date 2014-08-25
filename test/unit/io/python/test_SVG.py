#!/usr/bin/env py.test

"""Unit tests for SVG output"""

# Copyright (C) 2012 Anders Logg
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
# First added:  2012-12-01
# Last changed: 2012-12-01

import pytest
from dolfin import *

def test_write_mesh_1d():
    mesh = UnitIntervalMesh(8)
    f = File("output/mesh_1d.svg")
    f << mesh

def test_write_mesh_2d():
    mesh = UnitSquareMesh(8, 8)
    f = File("output/mesh_2d.svg")
    f << mesh

def test_write_mesh_3d():
    mesh = UnitCubeMesh(8, 8, 8)
    f = File("output/mesh_3d.svg")
    f << mesh
