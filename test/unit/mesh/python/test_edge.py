"""Unit tests for the Edge class"""

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
# First added:  2011-02-26
# Last changed: 2014-05-30

import pytest
from dolfin import *

cube   = UnitCubeMesh(5, 5, 5)
square = UnitSquareMesh(5, 5)
meshes = [cube, square]

skip_in_paralell = @pytest.mark.skipif(MPI.size(mpi_comm_world()) > 1, 
                           reason="Skipping unit test(s) not working in parallel")

class TestEdgeFunctions():

    @skip_in_paralell
    def test_2DEdgeLength(self):
        """Iterate over edges and sum length."""
        length = 0.0
        for e in edges(square):
            length += e.length()
        assert round(length - 19.07106781186544708362, 7) == 0

    @skip_in_paralell
    def test_3DEdgeLength(self):
        """Iterate over edges and sum length."""
        length = 0.0
        for e in edges(cube):
            length += e.length()
        assert round(length - 278.58049080280125053832, 7) == 0

    def test_EdgeDot(self):
        """Iterate over edges compute dot product with self."""
        for mesh in meshes:
            for e in edges(mesh):
                dot = e.dot(e)/(e.length()**2)
                assert round(dot - 1.0, 7) == 0

if __name__ == "__main__":
    pytest.main()
