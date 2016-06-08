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

import os
import pytest
from dolfin import *
from dolfin_utils.test import fixture, cd_tempdir

def test_write_mesh_1d(cd_tempdir):
    mesh = UnitIntervalMesh(8)
    f = File("_1d.svg")
    f << mesh

def test_write_mesh_2d(cd_tempdir):
    mesh = UnitSquareMesh(8, 8)
    f = File("2d.svg")
    f << mesh

def test_write_mesh_3d(cd_tempdir):
    mesh = UnitCubeMesh(8, 8, 8)
    f = File("3d.svg")
    f << mesh

def test_ipython_svg(cd_tempdir):
    # Test IPython SVG repr hooks
    mesh2d = UnitSquareMesh(8, 8)
    svg = mesh2d._repr_svg_()
    assert '<svg' in svg
    mesh3d = UnitCubeMesh(8, 8, 8)
    svg = mesh3d._repr_svg_()
    assert svg is None
