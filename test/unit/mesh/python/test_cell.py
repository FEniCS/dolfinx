"""Unit tests for the Cell class"""

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
# First added:  2013-04-18
# Last changed: 2014-05-30

import pytest
import numpy

from dolfin import *

skip_in_paralell = pytest.mark.skipif(MPI.size(mpi_comm_world()) > 1, 
                          reason="Skipping unit test(s) not working in parallel")

@skip_in_paralell
class IntervalTest():

    def test_distance(self):

        mesh = UnitIntervalMesh(1)
        cell = Cell(mesh, 0)

        assert round(cell.distance(Point(-1.0)) - 1.0, 7) == 0
        assert round(cell.distance(Point(0.5)) - 0.0, 7) == 0

@skip_in_paralell
class TriangleTest():

    def test_distance(self):

        mesh = UnitSquareMesh(1, 1)
        cell = Cell(mesh, 1)

        assert round(cell.distance(Point(-1.0, -1.0)) - numpy.sqrt(2), 7) == 0
        assert round(cell.distance(Point(-1.0, 0.5)) - 1, 7) == 0
        assert round(cell.distance(Point(0.5, 0.5)) - 0.0, 7) == 0

@skip_in_paralell
class TetrahedronTest():

    def test_distance(self):

        mesh = UnitCubeMesh(1, 1, 1)
        cell = Cell(mesh, 5)

        assert round(cell.distance(Point(-1.0, -1.0, -1.0)) - \
                      numpy.sqrt(3), 7) == 0
        assert round(cell.distance(Point(-1.0, 0.5, 0.5)) - 1, 7) == 0
        assert round(cell.distance(Point(0.5, 0.5, 0.5)) - 0.0, 7) == 0

if __name__ == "__main__":
        pytest.main()
