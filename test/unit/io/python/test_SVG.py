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

import os
import pytest
from dolfin import *

# create an output folder
@pytest.fixture(scope="module")
def temppath():
    filedir = os.path.dirname(os.path.abspath(__file__))
    basename = os.path.basename(__file__).replace(".py", "_data")
    temppath = os.path.join(filedir, basename, "")
    if not os.path.exists(temppath):
        os.mkdir(temppath)
    return temppath

def test_write_mesh_1d(temppath):
    mesh = UnitIntervalMesh(8)
    f = File(temppath + "_1d.svg")
    f << mesh

def test_write_mesh_2d(temppath):
    mesh = UnitSquareMesh(8, 8)
    f = File(temppath + "_2d.svg")
    f << mesh

def test_write_mesh_3d(temppath):
    mesh = UnitCubeMesh(8, 8, 8)
    f = File(temppath + "_3d.svg")
    f << mesh
