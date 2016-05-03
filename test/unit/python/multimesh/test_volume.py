#!/usr/bin/env py.test

"""Unit tests for multimesh volume computation"""

# Copyright (C) 2016 Anders Logg
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
# Last changed: 2016-05-03

from __future__ import print_function
import pytest

from dolfin import *
from dolfin_utils.test import skip_in_parallel

@skip_in_parallel
def test_volume_2d():
    "Integrate volume of union of 2D meshes"

    mesh_0 = UnitSquareMesh(8, 8)
    mesh_1 = UnitSquareMesh(8, 8)
