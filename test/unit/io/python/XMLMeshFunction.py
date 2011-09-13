"Unit tests for XML input/output of MeshFunction (class XMLMeshFunction)"

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
# Modified by Anders Logg 2011
#
# First added:  2011-09-13
# Last changed: 2011-09-13

import unittest
from dolfin import *

class XMLMeshFunction(unittest.TestCase):

    def test_io_uint(self):
        "Test input/output for uint"

        # Write some data
        mesh = UnitSquare(5, 5)
        f = MeshFunction("uint", mesh, 1)
        f.set_all(0)
        f[2] = 3
        f[5] = 7

        # Write
        output_file = File("XMLMeshFunction_test_io_uint.xml")
        output_file << f

        # Read from file
        g = MeshFunction("uint", mesh, 1)
        input_file = File("XMLMeshFunction_test_io_uint.xml")
        input_file >> g

        # Check values
        for i in xrange(f.size()):
            self.assertEqual(f[i], g[i])

    def test_io_int(self):
        "Test input/output for int"

        # Write some data
        mesh = UnitSquare(5, 5)
        f = MeshFunction("int", mesh, 1)
        f.set_all(0)
        f[2] = -3
        f[5] = 7

        # Write
        output_file = File("XMLMeshFunction_test_io_int.xml")
        output_file << f

        # Read from file
        g = MeshFunction("int", mesh, 1)
        input_file = File("XMLMeshFunction_test_io_int.xml")
        input_file >> g

        # Check values
        for i in xrange(f.size()):
            self.assertEqual(f[i], g[i])

    def test_io_double(self):
        "Test input/output for double"

        # Write some data
        mesh = UnitSquare(5, 5)
        f = MeshFunction("double", mesh, 1)
        f.set_all(0.0)
        f[2] = 3.14
        f[5] = 10000000.0

        # Write
        output_file = File("XMLMeshFunction_test_io_double.xml")
        output_file << f

        # Read from file
        g = MeshFunction("double", mesh, 1)
        input_file = File("XMLMeshFunction_test_io_double.xml")
        input_file >> g

        # Check values
        for i in xrange(f.size()):
            self.assertEqual(f[i], g[i])

    def test_io_bool(self):
        "Test input/output for bool"

        # Write some data
        mesh = UnitSquare(5, 5)
        f = MeshFunction("bool", mesh, 1)
        f.set_all(False)
        f[2] = True
        f[5] = False

        # Write
        output_file = File("XMLMeshFunction_test_io_bool.xml")
        output_file << f

        # Read from file
        g = MeshFunction("bool", mesh, 1)
        input_file = File("XMLMeshFunction_test_io_bool.xml")
        input_file >> g

        # Check values
        for i in xrange(f.size()):
            self.assertEqual(f[i], g[i])

if __name__ == "__main__":
    unittest.main()
