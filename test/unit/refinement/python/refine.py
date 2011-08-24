"""Unit tests for the refinement library"""

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
# First added:  2011-08-23
# Last changed:

import unittest
from dolfin import *

class MeshRefinement(unittest.TestCase):

    def test_uniform_refine2D(self):
        if MPI.num_processes() == 1:
            mesh = UnitSquare(4, 6)
            mesh = refine(mesh)

    def test_uniform_refine3D(self):
        if MPI.num_processes() == 1:
            mesh = UnitCube(4, 4, 6)
            mesh = refine(mesh)

if __name__ == "__main__":
    unittest.main()
