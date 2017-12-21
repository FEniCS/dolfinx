"""Unit tests for the solve interface"""

# Copyright (C) 2011 Garth N. Wells
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
# First added:  2011-12-21
# Last changed:se

from dolfin import *

def test_normalize_average():
    size = 200
    value = 2.0
    x = Vector(MPI.comm_world, size)
    x[:] = value
    factor = normalize(x, "average")
    assert factor == value
    assert x.sum() == 0.0

def test_normalize_l2():
    size = 200
    value = 2.0
    x = Vector(MPI.comm_world, size)
    x[:] = value
    factor = normalize(x, "l2")
    assert round(factor - sqrt(size*value*value), 7) == 0
    assert round(x.norm("l2") - 1.0, 7) == 0
