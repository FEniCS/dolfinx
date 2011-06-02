"""Unit tests for the Face class"""

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
# Last changed: 2011-02-26

import unittest
from dolfin import *

cube = UnitCube(5, 5, 5)

class Area(unittest.TestCase):

    def testArea(self):
        """Iterate over faces and sum area."""
        area = 0.0
        for f in faces(cube):
            area += f.area()
        if MPI.num_processes() == 1:
            self.assertAlmostEqual(area, 39.21320343559672494393)

class Normal(unittest.TestCase):

    def testNormalPoint(self):
        """Compute normal vector to each face."""
        for f in faces(cube):
            n = f.normal()
            self.assertAlmostEqual(n.norm(), 1.0)

    def testNormalComponent(self):
        """Compute normal vector components to each face."""
        D = cube.topology().dim()
        for f in faces(cube):
            n = [f.normal(i) for i in range(D)]
            norm = sum(map(lambda x: x*x, n))
            self.assertAlmostEqual(norm, 1.0)

if __name__ == "__main__":
    unittest.main()
