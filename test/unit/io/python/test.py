"""Unit tests for the io library"""

# Copyright (C) 2007 Anders Logg
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
# First added:  2009-01-02
# Last changed: 2009-01-02

import unittest
from dolfin import *

class VectorXML_IO(unittest.TestCase):

    def testSaveReadVector(self):
        size = 512
        x = Vector(size)
        x[:] = 1.0

        out_file = File("test_vector_xml.xml")
        out_file << x

        y = Vector()
        out_file >> y
        self.assertEqual(x.size(), y.size())
        self.assertAlmostEqual((x - y).norm("l2"), 0.0)


if __name__ == "__main__":
    unittest.main()
